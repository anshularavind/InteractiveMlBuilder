from celery import shared_task
from celery.signals import task_failure
from ..model_blocks import BuiltModel, train_model
from ..database import interface as db

@shared_task(bind=True, 
            max_retries=3, 
            soft_time_limit=3300,
            time_limit=3600)
def train_model_task(self, username, model_config, dataset):
    try:
        # Initialize progress tracking
        total_steps = 100
        self.update_state(state='PROGRESS', 
                         meta={
                             'current': 0,
                             'total': total_steps,
                             'status': 'Initializing training...'
                         })

        # Build the model
        model = BuiltModel(model_config)
        self.update_state(state='PROGRESS', 
                         meta={
                             'current': 10,
                             'total': total_steps,
                             'status': 'Model built successfully'
                         })

        # Train the model with progress updates
        training_result = train_model(model, dataset, 
                                    progress_callback=lambda p: self.update_state(
                                        state='PROGRESS',
                                        meta={
                                            'current': 10 + int(p * 80),
                                            'total': total_steps,
                                            'status': f'Training progress: {p*100:.2f}%'
                                        }
                                    ))

        # Save the model
        self.update_state(state='PROGRESS', 
                         meta={
                             'current': 90,
                             'total': total_steps,
                             'status': 'Saving model...'
                         })
        db.add_model(username, model, model_config)

        return {
            'current': 100,
            'total': total_steps,
            'status': 'Training completed successfully!',
            'result': training_result
        }

    except Exception as e:
        self.retry(exc=e, countdown=60)

@task_failure.connect
def handle_task_failure(task_id=None, exception=None, **kwargs):
    """Handle task failures globally"""
    error_msg = f"Task {task_id} failed: {str(exception)}"
    # Add your logging logic here
    print(error_msg)

def get_task_progress(task_id):
    """Retrieve task progress information"""
    task = train_model_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 100,
            'status': 'Task is waiting to start...'
        }
    elif task.state == 'PROGRESS':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 100),
            'status': task.info.get('status', '')
        }
    elif task.state == 'SUCCESS':
        response = {
            'state': task.state,
            'current': 100,
            'total': 100,
            'status': task.info.get('status', ''),
            'result': task.info.get('result', {})
        }
    else:
        response = {
            'state': task.state,
            'current': 0,
            'total': 100,
            'status': str(task.info)
        }
    return response