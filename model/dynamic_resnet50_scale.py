import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import timm
from timm.models.registry import register_model
from timm.models.layers import Mlp
import math
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

__all__ = [
    "dynamic_resnet50_scale",
]


# -----------------------------
# 1) Policy net (unchanged)
# -----------------------------
class policy_net_patch(nn.Module):
    def __init__(
        self,
        feature_in_chans,
        hidden_chans=128,
        kernel_size=7,
        feature_size=7,
        seq_len=4,
        is_policy_net=True,
    ):
        super().__init__()

        self.ln_list = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        partial(nn.LayerNorm, eps=1e-6)(1024),
                        partial(nn.LayerNorm, eps=1e-6)(512),
                    ]
                )
                for _ in range(seq_len)
            ]
        )
        self.act_layer = nn.GELU()
        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(
            in_channels=feature_in_chans,
            out_channels=feature_in_chans,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=feature_in_chans,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=feature_in_chans,
            out_channels=hidden_chans,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.linear1 = nn.Linear(hidden_chans * feature_size * feature_size, 1024, bias=False)
        self.linear2 = nn.Linear(1024, 512, bias=False)

        if is_policy_net:
            self.output_head = nn.Sequential(nn.Linear(512, 3, bias=False), nn.Sigmoid())
        else:
            self.output_head = nn.Sequential(nn.Linear(512, 1, bias=False))

    def forward(self, x, step_index):
        x = self.conv1(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        x = self.act_layer(x)
        x = self.flatten(x)

        x = self.linear1(x)
        x = self.ln_list[step_index][0](x)
        x = self.act_layer(x)

        x = self.linear2(x)
        x = self.ln_list[step_index][1](x)
        x = self.act_layer(x)

        x = self.output_head(x)
        return x


# -----------------------------
# 2) ResNet50 -> tokens backbone
# -----------------------------
class ResNet50TokenBackbone(nn.Module):
    """
    ResNet50 feature map -> token sequence:
      input:  (B,3,H,W)
      output: (B, L=Hf*Wf, C_out)
    stride=32 for resnet50 (Hf=H/32).
    """

    def __init__(self, img_size, pretrained=True, out_dim=None):
        super().__init__()
        self.img_size = img_size

        self.net = timm.create_model(
            "resnet50",
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,),
        )
        self.out_channels = self.net.feature_info.channels()[-1]  # 2048
        self.out_dim = out_dim if out_dim is not None else self.out_channels

        if self.out_dim != self.out_channels:
            self.proj = nn.Linear(self.out_channels, self.out_dim, bias=False)
        else:
            self.proj = None

    def forward(self, x):
        feats = self.net(x)[0]  # (B,2048,Hf,Wf)
        B, C, Hf, Wf = feats.shape
        tokens = feats.flatten(2).transpose(1, 2).contiguous()  # (B,L,2048)
        if self.proj is not None:
            tokens = self.proj(tokens)  # (B,L,out_dim)
        return tokens, (Hf, Wf)



# -----------------------------
# 3) Main model: ResNet50 backbone version
# -----------------------------
@register_model
def dynamic_resnet50_scale(**kwargs):
    return DynamicResNet50Scale(**kwargs)


class DynamicResNet50Scale(nn.Module):
    def __init__(
        self,
        seq_len,
        feature_in_chans,   # must match resnet out_channels (normally 2048)
        recover_n,
        remaining_blocks,   # kept for interface; resnet doesn't use transformer blocks
        glance_input_size,
        glance_net_depth, glance_net_mlp_ratio, glance_net_drop_path,  # unused but kept
        focus_patch_size,
        focus_net_reg_size,
        focus_net_depth, focus_net_mlp_ratio, focus_net_drop_path,     # unused but kept
        multi_cls_drop_path,
        policy_net_hidden_chans,
        policy_net_kernel_size,
        num_classes=1000,
        scale_min=0.2,
        scale_max=2.0,
        pretrained_backbone=True,
        **kwargs,
    ):
        super().__init__()

        self.seq_len = seq_len

        # ResNet50 stride = 32
        self.glance_input_size = glance_input_size
        # self.glance_feature_size = self.glance_input_size // 32

        self.focus_patch_size = focus_patch_size
        # self.focus_feature_size = self.focus_patch_size // 32

        def ceil_div(a, b):
            return (a + b - 1) // b

        self.glance_feature_size = ceil_div(self.glance_input_size, 32)
        self.focus_feature_size  = ceil_div(self.focus_patch_size, 32)


        self.feature_in_chans = feature_in_chans
        self.recover_n = recover_n
        self.remaining_blocks = remaining_blocks
        self.focus_net_reg_size = focus_net_reg_size

        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)

        # ---- backbones ----
        self.glance_net = ResNet50TokenBackbone(
            img_size=self.glance_input_size,
            pretrained=pretrained_backbone,
            out_dim=self.feature_in_chans
        )
        self.focus_net = ResNet50TokenBackbone(
            img_size=self.focus_patch_size,
            pretrained=pretrained_backbone,
            out_dim=self.feature_in_chans
        )

        # now tokens are already projected to feature_in_chans, so no assert needed
        # assert self.feature_in_chans == self.glance_net.out_dim


        # sanity: feature_in_chans must align
        # assert self.glance_net.out_channels == self.feature_in_chans, \
        #     f"feature_in_chans={feature_in_chans} != resnet_out={self.glance_net.out_channels}"

        # ---- classification head per step (replace DeiT remaining blocks + head) ----
        # Do: LN -> (optional MLP) -> mean pool over tokens -> linear
        self.post_fuse_norm = nn.ModuleList(
            [partial(nn.LayerNorm, eps=1e-6)(self.feature_in_chans) for _ in range(seq_len + 1)]
        )
        self.post_fuse_mlp_list = nn.ModuleList(
            [
                Mlp(
                    in_features=self.feature_in_chans,
                    hidden_features=int(self.feature_in_chans * 4),
                    act_layer=nn.GELU,
                    drop=0.0,
                )
                for _ in range(seq_len)
            ]
        )
        self.cls_head = nn.ModuleList(
            [nn.Linear(self.feature_in_chans, num_classes) for _ in range(seq_len + 1)]
        )

        # ---- policy nets (unchanged) ----
        self.policy_net_patch_norm_list_stateV = nn.ModuleList(
            [partial(nn.LayerNorm, eps=1e-6)(self.feature_in_chans) for _ in range(seq_len)]
        )
        self.policy_net_patch_norm_list = nn.ModuleList(
            [partial(nn.LayerNorm, eps=1e-6)(self.feature_in_chans) for _ in range(seq_len)]
        )

        self.policy_net_patch_stateV = policy_net_patch(
            feature_in_chans=self.feature_in_chans,
            hidden_chans=policy_net_hidden_chans,
            kernel_size=policy_net_kernel_size,
            feature_size=self.glance_feature_size,
            seq_len=self.seq_len,
            is_policy_net=False,
        )
        self.policy_net_patch = policy_net_patch(
            feature_in_chans=self.feature_in_chans,
            hidden_chans=policy_net_hidden_chans,
            kernel_size=policy_net_kernel_size,
            feature_size=self.glance_feature_size,
            seq_len=self.seq_len,
            is_policy_net=True,
        )

        # ---- offsets for recover (unchanged) ----
        n = self.recover_n
        offset_x = torch.arange(0, n) - (n - 1) // 2
        offset_y = torch.arange(0, n) - (n - 1) // 2
        offset_xy = torch.stack(torch.meshgrid([offset_x, offset_y], indexing="xy"))
        offset_xy = torch.flatten(offset_xy, -2).reshape(1, 2, 1, 1, n**2)
        self.register_buffer("offset_xy", offset_xy)

        # ---- reuse + recover heads (unchanged) ----
        self.policy_net_recover_reuseNorm_list = nn.ModuleList(
            [partial(nn.LayerNorm, eps=1e-6)(self.feature_in_chans) for _ in range(seq_len)]
        )
        self.policy_net_recover_reuseConvHead = nn.Conv2d(
            in_channels=self.feature_in_chans,
            out_channels=self.feature_in_chans,
            kernel_size=7,
            stride=1,
            padding=3,
            groups=self.feature_in_chans,
            bias=False,
        )

        self.policy_net_recover_reuseMLP_preLN_list = nn.ModuleList(
            [partial(nn.LayerNorm, eps=1e-6)(self.feature_in_chans) for _ in range(seq_len)]
        )
        self.policy_net_recover_reuseMLP = Mlp(
            in_features=self.feature_in_chans,
            hidden_features=self.feature_in_chans * 4,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.policy_net_recover_preLN_list = nn.ModuleList(
            [partial(nn.LayerNorm, eps=1e-6)(self.feature_in_chans) for _ in range(seq_len)]
        )
        self.policy_net_recover = Mlp(
            in_features=self.feature_in_chans,
            hidden_features=self.feature_in_chans * 4,
            out_features=self.recover_n**2,
            act_layer=nn.GELU,
            drop=0.0,
        )
        # focus-only (reg) classification head
        self.focus_only_norm = partial(nn.LayerNorm, eps=1e-6)(self.feature_in_chans)
        self.focus_only_head = nn.Linear(self.feature_in_chans, num_classes)

    # -----------------------------
    # interface
    # -----------------------------
    def forward(self, imgs=None, seq_l=4, old_states=None, old_actions=None, batch_index=None,
                flag="forward_backbone", ppo_std_this_iter=None):
        if flag == "forward_backbone":
            return self.forward_backbone(imgs, seq_l, ppo_std_this_iter=ppo_std_this_iter)
        elif flag == "evaluate_policy_net":
            return self.evaluate_policy_net(seq_l=seq_l, old_states=old_states, old_actions=old_actions,
                                            batch_index=batch_index, ppo_std_this_iter=ppo_std_this_iter)
        else:
            raise NotImplementedError

    def infer_focus_net_only(self, imgs):
        x = F.interpolate(imgs, size=(self.focus_patch_size, self.focus_patch_size), mode="bicubic")
        tokens, _ = self.focus_net(x)  # (B,L,C)
        logits = self.focus_only_head(self.focus_only_norm(tokens).mean(dim=1))  # (B,num_classes)
        return logits


    # -----------------------------
    # evaluate policy (unchanged)
    # -----------------------------
    def evaluate_policy_net(self, seq_l=None, old_states=None, old_actions=None, batch_index=None, ppo_std_this_iter=None):
        logprobs = []
        state_values = []
        dist_entropy = []

        for focus_step_index in range(seq_l):
            input_feat = old_states[focus_step_index][batch_index]  # (B,L,C)
            B, L, C = input_feat.shape

            with torch.cuda.amp.autocast(enabled=True):
                _state_values = self.policy_net_patch_stateV(
                    (
                        self.policy_net_patch_norm_list_stateV[focus_step_index](input_feat.detach())
                    ).permute(0, 2, 1).reshape(B, C, int(L**0.5), int(L**0.5)).contiguous(),
                    focus_step_index,
                )
                state_values.append(_state_values)

                _actions = self.policy_net_patch(
                    (
                        self.policy_net_patch_norm_list[focus_step_index](input_feat.detach())
                    ).permute(0, 2, 1).reshape(B, C, int(L**0.5), int(L**0.5)).contiguous(),
                    focus_step_index,
                )

                actions = _actions.clone()

                with torch.cuda.amp.autocast(enabled=False):
                    actions = actions.float()
                    action_var = torch.full((3,), ppo_std_this_iter, device=input_feat.device)
                    cov_mat = torch.diag(action_var)
                    dist = torch.distributions.multivariate_normal.MultivariateNormal(actions, scale_tril=cov_mat)

                    logprobs.append(dist.log_prob(old_actions[focus_step_index][batch_index]).unsqueeze(-1))
                    dist_entropy.append(dist.entropy().unsqueeze(-1))

        logprobs = torch.cat(logprobs, dim=1)
        state_values = torch.cat(state_values, dim=1)
        dist_entropy = torch.cat(dist_entropy, dim=1)
        return logprobs, state_values, dist_entropy

    # -----------------------------
    # forward backbone (resnet version)
    # -----------------------------
    def forward_backbone(self, imgs, seq_l=4, ppo_std_this_iter=None):
        expected_outputs = {
            "x_glance": [],
            "x_focus": [],
            "actions": [],
            "actions_logprobs": [],
            "_state_values": [],
            "states": [],
            "pos_std": [],
            "pos_mean": [],
            "scale_std": [],
            "scale_mean": [],
            "outputs_reg_focus_net": None,
        }
        if isinstance(imgs, (tuple, list)) and len(imgs) == 2:
            glance_imgs_gpu, full_imgs_cpu = imgs
        else:
            # fallback: old behavior (not recommended)
            glance_imgs_gpu = F.interpolate(imgs, size=(self.glance_input_size, self.glance_input_size), mode="bicubic")
            full_imgs_cpu = None

        expected_outputs["outputs_reg_focus_net"] = self.infer_focus_net_only(glance_imgs_gpu)


        # ---- glance backbone ----
        global_tokens, (Hf, Wf) = self.glance_net(glance_imgs_gpu)


        # classification at step0
        x0 = self.post_fuse_norm[0](global_tokens)
        x0 = x0.mean(dim=1)  # token avgpool
        x0 = self.cls_head[0](x0)
        expected_outputs["x_glance"].append(x0)

        updated_features = global_tokens

        for focus_step_index in range(seq_l):
            B, L, C = updated_features.shape

            with torch.cuda.amp.autocast(enabled=True):
                expected_outputs["states"].append(updated_features.detach())

                _state_values = self.policy_net_patch_stateV(
                    (
                        self.policy_net_patch_norm_list_stateV[focus_step_index](updated_features.detach())
                    ).permute(0, 2, 1).reshape(B, C, int(L**0.5), int(L**0.5)).contiguous(),
                    focus_step_index,
                )

                _actions = self.policy_net_patch(
                    (
                        self.policy_net_patch_norm_list[focus_step_index](updated_features.detach())
                    ).permute(0, 2, 1).reshape(B, C, int(L**0.5), int(L**0.5)).contiguous(),
                    focus_step_index,
                )

                actions = _actions.clone()

                if self.training:
                    with torch.cuda.amp.autocast(enabled=False):
                        actions = actions.float()
                        action_var = torch.full((3,), ppo_std_this_iter, device=updated_features.device)
                        cov_mat = torch.diag(action_var)
                        dist = torch.distributions.multivariate_normal.MultivariateNormal(actions, scale_tril=cov_mat)
                        actions = dist.sample()
                        actions = torch.clamp(actions, min=0, max=1)
                        actions_logprobs = dist.log_prob(actions)
                else:
                    actions_logprobs = torch.ones(B, device=updated_features.device)

                actions = actions.detach()
                expected_outputs["actions"].append(actions.detach())

                # (y,x,s) -> plus ones_like (kept as your original code)
                actions4 = torch.cat([actions, torch.ones_like(actions)], dim=1)

                # patches + two grids (int coords for neighbor, norm coords for grid_sample)
                # full_imgs_cpu must be provided in tuple mode
                assert full_imgs_cpu is not None, "CPU cropping requires full_imgs_cpu input."

                # CPU crop + build grids
                patches_cpu, grid_int, grid_norm = self.get_img_patches_cpu(
                    full_imgs_cpu, actions4, patch_size=self.focus_patch_size
                )
                # move grids to GPU (same device as updated_features/input_faet)
                grid_norm = grid_norm.to(updated_features.device, non_blocking=True)
                grid_int  = grid_int.to(updated_features.device, non_blocking=True)

                # move patch to GPU
                x_patch = patches_cpu.to(updated_features.device, non_blocking=True)

                feat_to_reuse = self.policy_net_recover_reuseMLP(
                    self.policy_net_recover_reuseMLP_preLN_list[focus_step_index](
                        self.get_reuse_faet(
                            input_faet=updated_features,
                            reuse_feat_grid=grid_norm,   # grid_sample needs [-1,1] grid
                            focus_step_index=focus_step_index,
                        )
                    )
                )

                # neighbor indexing needs pixel coords in [0, feature-1]
                neighbor_coords_index, outlier_mask = self.window_neighbor_grid(
                    B, grid_int, self.focus_feature_size, self.glance_feature_size
                )


                pos_std, pos_mean = torch.std_mean(actions[:, 0:2], dim=0)
                scale_std, scale_mean = torch.std_mean(actions[:, 2], dim=0)

            expected_outputs["actions_logprobs"].append(actions_logprobs.unsqueeze(-1))
            expected_outputs["_state_values"].append(_state_values)
            expected_outputs["pos_std"].append(pos_std)
            expected_outputs["pos_mean"].append(pos_mean)
            expected_outputs["scale_std"].append(scale_std)
            expected_outputs["scale_mean"].append(scale_mean)

            # ---- focus backbone ----
            # move patch to GPU (and normalize if你没在cpu里做)
            x_patch = patches_cpu.to(updated_features.device, non_blocking=True)

            # focus backbone (already patch_size x patch_size, so NO interpolate)
            local_tokens, _ = self.focus_net(x_patch)

            # fuse reuse context (simple additive; same shape (B,wHwW,C))
            if feat_to_reuse is not None and feat_to_reuse.shape == local_tokens.shape:
                local_tokens = local_tokens + feat_to_reuse

            B, wHwW, C = local_tokens.shape
            assert wHwW == self.focus_feature_size**2, f"expect {self.focus_feature_size**2}, got {wHwW}"

            with torch.cuda.amp.autocast(enabled=True):
                recover_policy_weight = self.policy_net_recover(
                    self.policy_net_recover_preLN_list[focus_step_index](local_tokens)
                )

                neighbor_coords_index, outlier_mask = self.window_neighbor_grid(
                    B, grid_int, self.focus_feature_size, self.glance_feature_size
                )


                recover_policy_weight = recover_policy_weight * outlier_mask

                relative_position_weight = torch.zeros(
                    (B, wHwW, self.glance_feature_size**2),
                    device=recover_policy_weight.device,
                    dtype=recover_policy_weight.dtype,
                ).scatter_add_(dim=2, index=neighbor_coords_index, src=recover_policy_weight).permute(0, 2, 1)

                updated_features = relative_position_weight @ local_tokens + updated_features

                # resnet version "post fuse block"
                updated_features = updated_features + self.post_fuse_mlp_list[focus_step_index](
                    self.post_fuse_norm[focus_step_index + 1](updated_features)
                )

            # classification at each step
            x_cls = self.post_fuse_norm[focus_step_index + 1](updated_features).mean(dim=1)
            x_cls = self.cls_head[focus_step_index + 1](x_cls)
            expected_outputs["x_focus"].append(x_cls)

        return expected_outputs

    # -----------------------------
    # reuse + recover utils (unchanged except stride sizes)
    # -----------------------------
    def get_reuse_faet(self, input_faet, reuse_feat_grid, focus_step_index):
        B, L, C = input_faet.shape
        input_faet = self.policy_net_recover_reuseNorm_list[focus_step_index](input_faet)
        input_faet = input_faet.permute(0, 2, 1).reshape(B, C, int(L**0.5), int(L**0.5)).contiguous()
        input_faet = self.policy_net_recover_reuseConvHead(input_faet)

        feat_to_reuse = F.grid_sample(input_faet, reuse_feat_grid, mode="bilinear")
        feat_to_reuse = feat_to_reuse.reshape(B, C, self.focus_feature_size**2).permute(0, 2, 1).contiguous()
        return feat_to_reuse

    def window_neighbor_grid(self, B, sub_grid, window_size, feature_size):
        assert sub_grid.dim() == 4 and sub_grid.size(1) == 2, \
            f"sub_grid must be (B,2,H,W), got {sub_grid.shape}"

        grid = sub_grid
        neighbor_coords = (grid + 0.5).int().unsqueeze(-1) + self.offset_xy

        neighbor_coords = neighbor_coords.reshape(B, 2, window_size * window_size * self.recover_n * self.recover_n)
        outlier_index = (
            (neighbor_coords[:, 0] >= 0)
            & (neighbor_coords[:, 1] >= 0)
            & (neighbor_coords[:, 0] <= (feature_size - 1))
            & (neighbor_coords[:, 1] <= (feature_size - 1))
        )
        outlier_mask = outlier_index.long().view(B, window_size * window_size, self.recover_n * self.recover_n)

        neighbor_coords = torch.clamp(neighbor_coords, min=0, max=(feature_size - 1))
        neighbor_coords_index = neighbor_coords[:, 0, :] + neighbor_coords[:, 1, :] * feature_size

        return neighbor_coords_index.view(B, window_size * window_size, self.recover_n * self.recover_n), outlier_mask

    def get_img_patches(self, input_frames, actions, patch_size, image_size):
        """
        Return:
        patches:    (B,3,patch_size,patch_size)
        grid_int:   (B,2,Hf,Wf)  in [0, feature_size-1] for neighbor indexing (x,y)
        grid_norm:  (B,Hf,Wf,2)  in [-1,1] for grid_sample
        """
        batchsize = actions.size(0)
        theta = torch.zeros((batchsize, 2, 3), device=input_frames.device)

        xy = actions[:, :2].clamp(0.0, 1.0)   # (B,2)  (y,x)
        s  = actions[:, 2:3].clamp(0.0, float(self.scale_max))  # (B,1)

        crop_mul = (float(self.scale_min) + s).clamp(min=1e-4)
        max_mul_by_image = (float(image_size) - 1.0) / float(patch_size)
        crop_mul = crop_mul.clamp(max=max_mul_by_image)

        patch_scale = torch.cat([crop_mul, crop_mul], dim=1)  # (B,2)

        patch_coordinate = xy * (float(image_size) - float(patch_size) * patch_scale)

        x1 = patch_coordinate[:, 1]
        y1 = patch_coordinate[:, 0]
        x2 = x1 + float(patch_size) * patch_scale[:, 1]
        y2 = y1 + float(patch_size) * patch_scale[:, 0]

        theta[:, 0, 0] = float(patch_size) * patch_scale[:, 1] / float(image_size)
        theta[:, 1, 1] = float(patch_size) * patch_scale[:, 0] / float(image_size)
        theta[:, 0, 2] = -1.0 + (x1 + x2) / float(image_size)
        theta[:, 1, 2] = -1.0 + (y1 + y2) / float(image_size)

        # crop image patch
        grid_patch = F.affine_grid(
            theta.float(),
            torch.Size((batchsize, 1, patch_size, patch_size)),
            align_corners=False
        )
        patches = F.grid_sample(input_frames, grid_patch, mode="bicubic", align_corners=False)

        # grid on feature resolution for reuse/recover
        grid_norm = F.affine_grid(
            theta.float(),
            torch.Size((batchsize, 1, self.focus_feature_size, self.focus_feature_size)),
            align_corners=False
        )  # (B,Hf,Wf,2) in [-1,1]

        # convert to [0, glance_feature_size-1] in pixel coords, then store as (B,2,Hf,Wf)
        grid_int = ((grid_norm + 1) / 2 * (self.glance_feature_size - 1)).permute(0, 3, 1, 2)

        return patches, grid_int, grid_norm

    @torch.no_grad()
    def cpu_crop_patches_lefttop(full_imgs_cpu, actions_cpu, patch_size, scale_min, scale_max, mean, std):
        """
        full_imgs_cpu: (B,3,H,W) float in [0,1] on CPU
        actions_cpu:   (B,3) (y,x,s) in [0,1] on CPU (detach().cpu())
        return: patches_cpu (B,3,patch_size,patch_size) normalized on CPU
        """
        B, _, H, W = full_imgs_cpu.shape
        patches = []

        yx = actions_cpu[:, 0:2].clamp(0.0, 1.0)    # (y,x)
        s  = actions_cpu[:, 2:3].clamp(0.0, 1.0)    # assume already [0,1]

        crop_mul = (scale_min + s).clamp(min=1e-4)  # same as your code
        # keep crop within image
        max_mul_h = (H - 1.0) / float(patch_size)
        max_mul_w = (W - 1.0) / float(patch_size)
        crop_mul = torch.min(crop_mul, torch.tensor([[min(max_mul_h, max_mul_w)]], dtype=crop_mul.dtype))

        crop_h = (float(patch_size) * crop_mul).squeeze(1)  # (B,)
        crop_w = (float(patch_size) * crop_mul).squeeze(1)

        # left-top in pixels
        y1 = yx[:, 0] * (float(H) - crop_h)
        x1 = yx[:, 1] * (float(W) - crop_w)

        for i in range(B):
            top  = int(torch.floor(y1[i]).item())
            left = int(torch.floor(x1[i]).item())
            h    = int(torch.ceil(crop_h[i]).item())
            w    = int(torch.ceil(crop_w[i]).item())

            # safety clamp
            top  = max(0, min(top, H - 1))
            left = max(0, min(left, W - 1))
            h    = max(1, min(h, H - top))
            w    = max(1, min(w, W - left))

            # crop then resize to patch_size
            patch = TF.resized_crop(
                full_imgs_cpu[i], top=top, left=left, height=h, width=w,
                size=[patch_size, patch_size],
                interpolation=InterpolationMode.BICUBIC
            )
            patch = TF.normalize(patch, mean=mean, std=std)
            patches.append(patch)

        return torch.stack(patches, dim=0)  # CPU tensor

    @torch.no_grad()
    def get_img_patches_cpu(self, full_imgs_cpu, actions, patch_size):
        """
        full_imgs_cpu: (B,3,H,W) float32 on CPU, range [0,1] (unnormalized)
        actions: (B,4) on GPU or CPU, but will be used as CPU here
                actions[:, :3] = (y, x, s_norm) in [0,1], y/x are LEFT-TOP
        Return:
        patches_cpu: (B,3,patch_size,patch_size)  (NOT normalized;你可选择这里normalize)
        grid_int:    (B,2,Hf,Wf)  in [0, glance_feature_size-1]  (for neighbor indexing)
        grid_norm:   (B,Hf,Wf,2)  in [-1,1]  (for grid_sample reuse)
        """
        if full_imgs_cpu.device.type != "cpu":
            full_imgs_cpu = full_imgs_cpu.cpu()

        # actions -> CPU float
        if actions.device.type != "cpu":
            a = actions[:, :3].detach().float().cpu()
        else:
            a = actions[:, :3].detach().float()

        B, _, H, W = full_imgs_cpu.shape

        yx = a[:, 0:2].clamp(0.0, 1.0)      # (y,x)
        s  = a[:, 2:3].clamp(0.0, 1.0)      # s_norm

        crop_mul = (self.scale_min + s).clamp(min=1e-4)

        # keep crop inside image
        max_mul_h = (H - 1.0) / float(patch_size)
        max_mul_w = (W - 1.0) / float(patch_size)
        max_mul = min(max_mul_h, max_mul_w)
        crop_mul = torch.clamp(crop_mul, max=max_mul)

        crop_h = (float(patch_size) * crop_mul).squeeze(1)  # (B,)
        crop_w = (float(patch_size) * crop_mul).squeeze(1)

        y1 = yx[:, 0] * (float(H) - crop_h)
        x1 = yx[:, 1] * (float(W) - crop_w)

        patches = []
        for i in range(B):
            top  = int(torch.floor(y1[i]).item())
            left = int(torch.floor(x1[i]).item())
            h    = int(torch.ceil(crop_h[i]).item())
            w    = int(torch.ceil(crop_w[i]).item())

            top  = max(0, min(top, H - 1))
            left = max(0, min(left, W - 1))
            h    = max(1, min(h, H - top))
            w    = max(1, min(w, W - left))

            patch = TF.resized_crop(
                full_imgs_cpu[i],
                top=top, left=left, height=h, width=w,
                size=[patch_size, patch_size],
                interpolation=InterpolationMode.BICUBIC
            )
            patches.append(patch)

        patches_cpu = torch.stack(patches, dim=0)  # (B,3,patch,patch)

        # ---- build reuse/recover grids (NO dependence on full image pixels) ----
        # We re-create theta the same way as your get_img_patches (but image_size is H/W here).
        # Note: assumes square W==H; if not square, you should compute separately.
        image_size = float(W)

        theta = torch.zeros((B, 2, 3), dtype=torch.float32)
        patch_scale = torch.cat([crop_mul, crop_mul], dim=1)  # (B,2)

        patch_coordinate = yx * (image_size - float(patch_size) * patch_scale)  # (y1,x1) in pixels
        x1p = patch_coordinate[:, 1]
        y1p = patch_coordinate[:, 0]
        x2p = x1p + float(patch_size) * patch_scale[:, 1]
        y2p = y1p + float(patch_size) * patch_scale[:, 0]

        theta[:, 0, 0] = float(patch_size) * patch_scale[:, 1] / image_size
        theta[:, 1, 1] = float(patch_size) * patch_scale[:, 0] / image_size
        theta[:, 0, 2] = -1.0 + (x1p + x2p) / image_size
        theta[:, 1, 2] = -1.0 + (y1p + y2p) / image_size

        # grid on feature resolution for reuse/recover (B,Hf,Wf,2) in [-1,1]
        grid_norm = F.affine_grid(
            theta,
            torch.Size((B, 1, self.focus_feature_size, self.focus_feature_size)),
            align_corners=False
        )

        # convert to integer coords on glance feature map (B,2,Hf,Wf)
        grid_int = ((grid_norm + 1) / 2 * (self.glance_feature_size - 1)).permute(0, 3, 1, 2)

        return patches_cpu, grid_int, grid_norm
