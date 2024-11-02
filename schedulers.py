# Schedulers for our model
import optax

def get_adam(params):
    optimizer = optax.adam(learning_rate = 1e-5)
    opt_state = optimizer.init(params)
    return optimizer, opt_state

def get_adan(params):
    learning_rate = 5e-5
    weight_decay = 1e-4
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    optimizer = optax.adan(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        b1=beta1,
        b2=beta2,
        eps=epsilon
    )

    opt_state = optimizer.init(params)
    return optimizer, opt_state


def get_onecycle(params, epoch):

    if epoch == 0:
        transition_steps = 2e6
        peak_value = 1e-3
    elif epoch == 1:
        transition_steps = 1e6
        peak_value = 1e-4
    elif epoch == 2:
        transition_steps = 500e3
        peak_value = 1e-5

    print(f'{transition_steps=}')
    print(f'{peak_value=}')
    one_cycle_schedule = optax.schedules.cosine_onecycle_schedule(
        transition_steps = transition_steps,
        peak_value = peak_value,
        pct_start = 0.1,
        div_factor = 10,
        final_div_factor = 100
    )

    optimizer = optax.adamw(
        learning_rate=one_cycle_schedule,
        weight_decay=1e-4, 
        b1=0.9,
        b2=0.999,
        eps=1e-8
    )
    opt_state = optimizer.init(params)
    return optimizer, opt_state

