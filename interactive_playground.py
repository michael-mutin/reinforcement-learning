from grid import Environment, Action
import keyboard

# Example
env = Environment(7, (0,0))
env.createHole((1,1))
env.createHole((1,2))
env.createHole((5,3))
env.createHole((4,6))
env.createHole((0,3))
next_state = (6,6)

def print_next_state(next_state, reward, done):
    print('\033[2J\033[H', end='')
    env.view_grid(next_state)
    print(f"Reward: {reward}, Done: {done}")

def on_key_event(event):
    global next_state
    if event.event_type == 'down':
        match event.scan_code:
            case 72: # up                
                next_state, reward, done = env.step(next_state, Action.UP)
                print_next_state(next_state, reward, done)
            case 80: # down
                next_state, reward, done = env.step(next_state, Action.DOWN)
                print_next_state(next_state, reward, done)
            case 75: # left
                next_state, reward, done = env.step(next_state, Action.LEFT)
                print_next_state(next_state, reward, done)
            case 77: # right
                next_state, reward, done = env.step(next_state, Action.RIGHT)
                print_next_state(next_state, reward, done)

keyboard.on_press(on_key_event)
print("Press ESC to exit the running program.")
env.view_grid(next_state)
keyboard.wait('esc')