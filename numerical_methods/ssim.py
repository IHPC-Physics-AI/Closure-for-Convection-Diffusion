import jax
import jax.numpy as jnp

def gaussian_2d(size, sigma):
  gauss = jnp.array([jnp.exp(-(x - size//2)**2/float(2*sigma**2)) for x in range(size)])
  kernel = jnp.outer(gauss, gauss)
  return kernel/jnp.sum(kernel)

def ssim(preds: jax.Array, 
         target: jax.Array, 
         window_size: int,
         sigma: float = 1.5,
         k1: float = 0.01,
         k2: float = 0.03):
    """Adapted from SSIM implementation of PyTorch

    Args:
        preds (jax.Array): 
            Array of dimension 2 of the predicted image
        target (jax.Array):
            Array of dimension 2 of the target image
        k1 (float):
            Used to compute constant to prevent numerical stability.
            Defaults to 0.01.
        k2 (float): 
            Used to compute constant to prevent numerical stability.
            Defaults to 0.03.
    """
    data_range = jnp.max(jnp.array((preds.max()-preds.min(), target.max()-target.min())))
    c1 = jnp.pow(k1 * data_range, 2)  
    c2 = jnp.pow(k2 * data_range, 2)
    
    kernel = gaussian_2d(size=window_size, sigma=sigma)
    
    preds_kernel = jax.scipy.signal.convolve2d(preds, kernel, mode='same')
    target_kernel = jax.scipy.signal.convolve2d(target, kernel, mode='same')
    preds_preds_kernel = jax.scipy.signal.convolve2d(preds*preds, kernel, mode='same')
    target_target_kernel = jax.scipy.signal.convolve2d(target*target, kernel, mode='same')
    preds_target_kernel = jax.scipy.signal.convolve2d(preds*target, kernel, mode='same')
    
    mu_pred_sq = jnp.pow(preds_kernel, 2)
    mu_target_sq = jnp.pow(target_kernel, 2)
    mu_pred_target = preds_kernel * target_kernel
    
    sigma_pred_sq = jnp.clip(preds_preds_kernel - mu_pred_sq, min=0.0)
    sigma_target_sq = jnp.clip(target_target_kernel - mu_target_sq, min=0.0)
    sigma_pred_target = preds_target_kernel - mu_pred_target
    
    upper = 2 * sigma_pred_target + c2
    lower = (sigma_pred_sq + sigma_target_sq) + c2

    ssim_idx_full_image = ((2 * mu_pred_target + c1) * upper) / ((mu_pred_sq + mu_target_sq + c1) * lower)
    contrast_sensitivity = upper / lower
    
    return jnp.mean(ssim_idx_full_image), jnp.mean(contrast_sensitivity)

def multiscale_ssim(preds: jax.Array,
                    target: jax.Array,
                    betas: tuple):
    mcs_list = []
    for m in range(len(betas)):
        h,w = preds.shape
        window_size = min(11, h, w) if min(h, w) >= 11 else max(3, min(h, w))
        sim, contrast_sensitivity = ssim(
            preds=preds, target=target, window_size=window_size, sigma=1.5, k1=0.01, k2=0.03
        )
        mcs_list.append(contrast_sensitivity)
        preds = jax.image.resize(preds, shape=(h//2, w//2), method="linear")
        target = jax.image.resize(target, shape=(h//2, w//2), method="linear")
    
    mcs_list[-1] = sim
    
    mcs_weighted = jnp.array(mcs_list) ** jnp.array(betas)
    return jnp.prod(mcs_weighted)
