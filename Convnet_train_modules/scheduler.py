from transformers import ( 
    get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)

def init_scheduler(lr_scheduler_type, optimizer, num_training_steps, num_warmup_steps):
    schedule_func = SCHEDULER_TYPE_TO_FUNCTION[lr_scheduler_type]
    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    

SCHEDULER_TYPE_TO_FUNCTION = {
    'linear': get_linear_schedule_with_warmup,
    'cosine': get_cosine_schedule_with_warmup,
    'cosine_with_restarts': get_cosine_with_hard_restarts_schedule_with_warmup,
    'polynomial': get_polynomial_decay_schedule_with_warmup,
    'constant': get_constant_schedule_with_warmup
}