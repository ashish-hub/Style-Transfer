import compute_grads
import compute_losses
import content_loss
import content_style_layers
import get_feature_representations
import get_model
import run_style_transfer
import show_images
import style_loss

best, best_loss = run_style_transfer(content_path,
                                     style_path,
                                     verbose=True,
                          show_intermediates=True)
