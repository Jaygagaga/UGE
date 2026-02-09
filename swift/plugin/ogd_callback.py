"""
OGD Callback for Swift Training

This callback handles OGD memory updates from pretrained checkpoints.
"""

from transformers import TrainerCallback
from swift.utils import get_logger

logger = get_logger()


class OGDCallback(TrainerCallback):
    """Callback to update OGD memory from pretrained model before training starts."""
    
    def __init__(self):
        """Initialize the callback."""
        self.ogd_trainer = None
    
    def _get_ogd_trainer(self, args):
        """Get OGD trainer from args."""
        # Check args directly first
        if hasattr(args, '_ogd_trainer') and args._ogd_trainer is not None:
            return args._ogd_trainer
        
        # Check training_args if it exists
        if hasattr(args, 'training_args') and hasattr(args.training_args, '_ogd_trainer') and args.training_args._ogd_trainer is not None:
            return args.training_args._ogd_trainer
        
        # Check trainer.args if trainer is available (set by HuggingFace Trainer)
        if hasattr(self, 'trainer') and self.trainer is not None and hasattr(self.trainer, 'args'):
            trainer_args = self.trainer.args
            if hasattr(trainer_args, '_ogd_trainer') and trainer_args._ogd_trainer is not None:
                return trainer_args._ogd_trainer
        return None
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Update OGD memory when training begins."""
        # Check if OGD is enabled (check both args and training_args)
        ogd_trainer = self._get_ogd_trainer(args)
        if ogd_trainer is None:
            # OGD trainer might not be created yet (created in create_optimizer_and_scheduler)
            # Log this for debugging
            print("[OGD] on_train_begin called but OGD trainer not found yet (will be created during optimizer initialization)", flush=True)
            logger.info("[OGD] on_train_begin called but OGD trainer not found yet (will be created during optimizer initialization)")
            return
        
        # Store reference for later use
        self.ogd_trainer = ogd_trainer
        
        # Check if memory has already been updated
        if hasattr(ogd_trainer.memory, '_memory_updated') and ogd_trainer.memory._memory_updated:
            logger.info("[OGD] Memory already updated, skipping...")
            return
        
        logger.info("[OGD] Updating memory from pretrained model...")
        
        # Get validation dataset for memory update
        # We use validation data to avoid overfitting to training data
        eval_dataset = kwargs.get('eval_dataset')
        train_dataset = kwargs.get('train_dataset')
        
        # If not in kwargs, try to get from trainer
        if eval_dataset is None and hasattr(self, 'trainer') and self.trainer is not None:
            eval_dataset = getattr(self.trainer, 'eval_dataset', None)
        if train_dataset is None and hasattr(self, 'trainer') and self.trainer is not None:
            train_dataset = getattr(self.trainer, 'train_dataset', None)
        
        # Prefer validation dataset, fallback to training dataset
        dataset_for_memory = eval_dataset if eval_dataset is not None else train_dataset
        
        if dataset_for_memory is None:
            logger.warning("[OGD] No dataset available for memory update! OGD will not work properly.")
            return
        
        # Create a simple dataloader for memory update
        from torch.utils.data import DataLoader
        
        # Use the trainer's data collator to properly handle variable-length sequences
        # (e.g., variable-length graphs, images, etc.)
        data_collator = None
        if hasattr(self, 'trainer') and self.trainer is not None:
            data_collator = getattr(self.trainer, 'data_collator', None)
        if data_collator is None:
            logger.warning("[OGD] No data_collator found in trainer, using default collate (may fail with variable-length sequences)")
        
        # Use a small batch size for memory update
        batch_size = min(32, getattr(args, 'per_device_eval_batch_size', 32))
        
        # Create dataloader with proper collation
        memory_dataloader = DataLoader(
            dataset_for_memory,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle for reproducibility
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=False,
            collate_fn=data_collator  # Use trainer's collator to handle variable-length sequences
        )
        
        # Update memory
        # Try to get num_batches from various sources
        num_batches = 100  # default
        if hasattr(args, 'ogd_update_memory_batches'):
            num_batches = args.ogd_update_memory_batches
        elif hasattr(self, 'trainer') and self.trainer is not None and hasattr(self.trainer, 'args'):
            trainer_args = self.trainer.args
            if hasattr(trainer_args, 'ogd_update_memory_batches'):
                num_batches = trainer_args.ogd_update_memory_batches
        # Also check the original args object if available
        if hasattr(self, 'trainer') and self.trainer is not None and hasattr(self.trainer, 'model') and hasattr(self.trainer.model, 'config'):
            # Try to get from model's training args if stored there
            pass  # This is handled above
        
        # For gradient-based OGD, we need a loss function
        # Try to get loss function from trainer if available
        loss_fn = None
        if hasattr(self, 'trainer') and self.trainer is not None:
            if hasattr(self.trainer, 'compute_loss'):
                loss_fn = self.trainer.compute_loss
            elif hasattr(self.trainer, 'loss_fn'):
                loss_fn = self.trainer.loss_fn
        
        ogd_trainer.update_memory(memory_dataloader, num_batches=num_batches, loss_fn=loss_fn)
        
        # Mark memory as updated
        ogd_trainer.memory._memory_updated = True
        
        # Enable projection for training
        ogd_trainer.enable_projection()
        
        # Set output directory for saving stats
        # Try multiple ways to get output_dir
        output_dir = None
        
        # Method 1: Direct from args
        if hasattr(args, 'output_dir'):
            output_dir = args.output_dir
        
        # Method 2: From training_args (nested in args)
        if output_dir is None and hasattr(args, 'training_args'):
            training_args = args.training_args
            if hasattr(training_args, 'output_dir'):
                output_dir = training_args.output_dir
        
        # Method 3: From trainer.args
        if output_dir is None and hasattr(self, 'trainer') and self.trainer is not None:
            if hasattr(self.trainer, 'args'):
                trainer_args = self.trainer.args
                if hasattr(trainer_args, 'output_dir'):
                    output_dir = trainer_args.output_dir
                # Also check training_args nested in trainer.args
                if output_dir is None and hasattr(trainer_args, 'training_args'):
                    if hasattr(trainer_args.training_args, 'output_dir'):
                        output_dir = trainer_args.training_args.output_dir
        
        if output_dir:
            import os
            # Convert to absolute path if relative
            output_dir = os.path.abspath(os.path.expanduser(str(output_dir)))
            ogd_trainer.set_output_dir(output_dir)
            logger.info(f"[OGD] Will save projection stats to: {output_dir}")
            
            # Verify directory exists and is writable
            try:
                os.makedirs(output_dir, exist_ok=True)
                # Test write
                test_file = os.path.join(output_dir, '.ogd_test_write')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                logger.info(f"[OGD] Verified output directory is writable: {output_dir}")
            except Exception as e:
                logger.error(f"[OGD] ERROR: Cannot write to output directory {output_dir}: {e}")
        else:
            logger.warning("[OGD] No output_dir found, projection stats will not be saved to file")
            logger.warning(f"[OGD] Debug: args type={type(args)}, has output_dir={hasattr(args, 'output_dir')}")
            if hasattr(self, 'trainer') and self.trainer is not None:
                logger.warning(f"[OGD] Debug: trainer.args type={type(self.trainer.args) if hasattr(self.trainer, 'args') else None}")
        
        logger.info("[OGD] Memory update complete. OGD projection enabled for training.")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Inject OGD projection stats into training logs.
        
        This is called at each logging step, allowing us to add OGD metrics
        to the training logs that appear in logging.jsonl. We modify the logs
        dict directly so the stats appear in the same log entry as loss, margin, etc.
        """
        if logs is None:
            return
        
        # Get OGD trainer
        ogd_trainer = self._get_ogd_trainer(args)
        if ogd_trainer is None or not ogd_trainer.projection_enabled:
            return
        
        # Get aggregated projection stats
        stats = ogd_trainer.get_projection_stats()
        projection_count = stats.get('ogd_projection_count', 0)
        
        # Only add stats if we have actual projection data
        # This prevents adding zeros when OGD hasn't projected anything yet
        if projection_count > 0:
            # Add stats directly to logs dict - this will appear in the same log entry
            logs.update(stats)
            logger.debug(
                f"[OGD] Added projection stats to logs: mean={stats.get('ogd_projection_mean', 0):.4f}, "
                f"count={stats.get('ogd_projection_count', 0)}, layers={stats.get('ogd_projection_layers', 0)}"
            )
            print(f"[OGD] Added projection stats to logs: mean={stats.get('ogd_projection_mean', 0):.4f}, "
                f"count={stats.get('ogd_projection_count', 0)}, layers={stats.get('ogd_projection_layers', 0)}"
            )
            
            # Save stats to file before resetting
            try:
                ogd_trainer.save_projection_stats(state.global_step, force_save=False)
            except Exception as e:
                logger.error(f"[OGD] ERROR saving projection stats at step {state.global_step}: {e}", exc_info=True)
            
            # Reset stats for next logging interval
            ogd_trainer.reset_projection_stats()
        else:
            # Log when stats are empty (for debugging)
            # Check if output_dir is set
            output_dir_status = "set" if ogd_trainer._output_dir else "NOT SET"
            if state.global_step % (args.logging_steps * 5) == 0:  # Only log occasionally
                logger.warning(
                    f"[OGD] No projection stats available at step {state.global_step}. "
                    f"Projection enabled: {ogd_trainer.projection_enabled}, "
                    f"Protected layers: {len(ogd_trainer.protected_layers)}, "
                    f"Output dir: {output_dir_status}"
                )
                print(f"[OGD] No projection stats available at step {state.global_step}. "
                    f"Projection enabled: {ogd_trainer.projection_enabled}, "
                    f"Protected layers: {len(ogd_trainer.protected_layers)}, "
                    f"Output dir: {output_dir_status}")
    
    def on_save(self, args, state, control, **kwargs):
        """Save projection stats when checkpoints are saved."""
        import torch.distributed as dist
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        
        logger.info(f"Rank {rank}: [OGD] on_save callback called")
        ogd_trainer = self._get_ogd_trainer(args)
        if ogd_trainer is None or not ogd_trainer.projection_enabled:
            logger.info(f"Rank {rank}: [OGD] OGD not enabled, skipping")
            return
        
        # Ensure output_dir is set (might not have been set in on_train_begin)
        if ogd_trainer._output_dir is None:
            # Try to get output_dir again
            output_dir = getattr(args, 'output_dir', None)
            if output_dir is None and hasattr(args, 'training_args'):
                output_dir = getattr(args.training_args, 'output_dir', None)
            if output_dir is None and hasattr(self, 'trainer') and self.trainer is not None:
                if hasattr(self.trainer, 'args'):
                    output_dir = getattr(self.trainer.args, 'output_dir', None)
            if output_dir:
                import os
                output_dir = os.path.abspath(os.path.expanduser(str(output_dir)))
                ogd_trainer.set_output_dir(output_dir)
                logger.info(f"Rank {rank}: [OGD] Set output_dir in on_save: {output_dir}")
        
        # Save stats when checkpoint is saved
        if state is not None:
            global_step = getattr(state, 'global_step', None)
            if global_step is not None:
                try:
                    logger.info(f"Rank {rank}: [OGD] About to save projection stats (step {global_step})...")
                    ogd_trainer.save_projection_stats(global_step, force_save=False)
                    logger.info(f"Rank {rank}: [OGD] ✅ Saved projection stats at checkpoint save (step {global_step})")
                except Exception as e:
                    logger.error(f"Rank {rank}: [OGD] ERROR saving projection stats at checkpoint (step {global_step}): {e}", exc_info=True)
        else:
            logger.info(f"Rank {rank}: [OGD] State is None, skipping projection stats save")
    
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Cleanup OGD hooks when training ends."""
        ogd_trainer = self._get_ogd_trainer(args)
        
        if ogd_trainer is not None:
            # Save final stats before cleanup
            if state is not None:
                final_step = getattr(state, 'global_step', None)
                if final_step is not None:
                    try:
                        ogd_trainer.save_projection_stats(final_step, force_save=True)
                        logger.info(f"[OGD] Saved final projection stats at step {final_step}")
                    except Exception as e:
                        logger.error(f"[OGD] ERROR saving final projection stats at step {final_step}: {e}", exc_info=True)
            
            ogd_trainer.remove_hooks()
            logger.info("[OGD] Removed hooks and cleaned up.")

