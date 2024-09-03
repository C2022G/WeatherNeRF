import torch.optim.lr_scheduler
import vren


# class Distortion(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, ws, deltas, ts, rays_a):
#         loss, ws_inclusive_scan, wts_inclusive_scan = \
#             vren.distortion_loss_fw(ws, deltas, ts, rays_a)
#         ctx.save_for_backward(ws_inclusive_scan, wts_inclusive_scan,
#                               ws, deltas, ts, rays_a)
#         return loss
#
#     @staticmethod
#     def backward(ctx, dL_dloss):
#         (ws_inclusive_scan, wts_inclusive_scan,
#          ws, deltas, ts, rays_a) = ctx.saved_tensors
#         dL_dws = vren.distortion_loss_bw(dL_dloss, ws_inclusive_scan,
#                                          wts_inclusive_scan,
#                                          ws, deltas, ts, rays_a)
#         return dL_dws, None, None, None

class NeDepth(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sigmas, deltas, ts, rays_a, vr_samples):
        ne_depth = \
            vren.ne_depth_fw(sigmas, deltas, ts, rays_a, vr_samples)
        ctx.save_for_backward(ne_depth, sigmas, deltas, ts, rays_a, vr_samples)
        return ne_depth

    @staticmethod
    def backward(ctx, dL_ddepth):
        (ne_depth, sigmas, deltas, ts, rays_a, vr_samples) = ctx.saved_tensors
        dl_dsigmas = vren.ne_depth_bw(dL_ddepth, ne_depth, sigmas, deltas, ts,
                                      rays_a, vr_samples)
        return dl_dsigmas, None, None, None, None
