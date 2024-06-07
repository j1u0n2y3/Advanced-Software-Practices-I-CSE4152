import argparse
import matplotlib.pyplot as plt

def p(current_error, Kp):
    return Kp * current_error

def simulate_p(Kp, setpoint, initial_position, initial_velocity, dt, num_steps):
    positions, velocities, errors = [initial_position], [initial_velocity], [setpoint - initial_position]

    for _ in range(num_steps):
        current_error = setpoint - positions[-1]

        control_output = p(current_error, Kp)

        velocities.append(velocities[-1] + (control_output * dt))
        positions.append(positions[-1] + (velocities[-1] * dt))

        errors.append(setpoint - positions[-1])

    return positions, velocities


def pd(current_error, previous_error, Kp, Kd, dt):
    return (Kp * current_error) + (Kd * (current_error - previous_error) / dt)

def simulate_pd(Kp, Kd, setpoint, initial_position, initial_velocity, dt, num_steps):
    positions, velocities, errors = [initial_position], [initial_velocity], [setpoint - initial_position]
    previous_error = errors[0]

    for _ in range(num_steps):
        current_error = setpoint - positions[-1]

        control_output = pd(current_error, previous_error, Kp, Kd, dt)

        next_velocity = velocities[-1] + control_output * dt
        next_position = positions[-1] + next_velocity * dt

        velocities.append(next_velocity)
        positions.append(next_position)

        errors.append(setpoint - next_position)
        previous_error = current_error

    return positions, velocities


def bb(current_value, setpoint, tolerance, max_output):
    return max_output if current_value < setpoint - tolerance else 0.0

def simulate_bb(setpoint, initial_value, tolerance, max_output, max_acceleration, dt, num_steps, minus=0.5):
    values, velocities = [initial_value], [0.0]

    for _ in range(num_steps):
        current_value, current_velocity = values[-1], velocities[-1]

        control_output = bb(current_velocity, 5.0, tolerance, max_output)

        next_velocity = max(0, current_velocity + (control_output - minus) * dt)

        values.append(current_value + next_velocity * dt)
        velocities.append(next_velocity)

    return values, velocities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', action='store_true')
    parser.add_argument('--pd', action='store_true')
    parser.add_argument('--bb', action='store_true')
    args = parser.parse_args()

    if args.p:
        Kp = 0.05
        setpoint, initial_position, initial_velocity = 10.0, 0.0, 0.0
        dt, num_steps = 0.1, 2000

        system_positions, system_velocities = simulate_p(
            Kp, setpoint, initial_position, initial_velocity, dt, num_steps
        )
        time_steps = [step * dt for step in range(num_steps + 1)]

        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, system_positions, 'o-', label='System State')
        plt.axhline(y=setpoint, color='r', linestyle='--', label='Setpoint')
        plt.title('P Control Simulation')
        plt.xlabel('Time (s)')
        plt.ylabel('State')
        print("P", end="")

    elif args.pd:
        Kp, Kd = 0.05, 0.1
        setpoint, initial_position, initial_velocity = 10.0, 0.0, 0.0
        dt, num_steps = 0.1, 2000

        system_positions, system_velocities = simulate_pd(
            Kp, Kd, setpoint, initial_position, initial_velocity, dt, num_steps
        )
        time_steps = [step * dt for step in range(num_steps + 1)]

        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, system_positions, 'o-', label='System State')
        plt.axhline(y=setpoint, color='r', linestyle='--', label='Setpoint')
        plt.title('PD Control Simulation')
        plt.xlabel('Time (s)')
        plt.ylabel('State')
        print("PD", end="")

    elif args.bb:
        setpoint, initial_value = 20.0, 0.0
        tolerance, max_output, max_acceleration = 0.5, 5.0, 2.0
        dt, num_steps = 0.1, 200
        minus = 0.1

        system_values, system_velocities = simulate_bb(
            setpoint, initial_value, tolerance, max_output, max_acceleration, dt, num_steps, minus
        )
        time_steps = [step * dt for step in range(num_steps + 1)]

        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, system_velocities, 'o-', label='System State')
        plt.axhline(y=5.0, color='r', linestyle='--', label='Setpoint')
        plt.title('BB Control Simulation')
        plt.xlabel('Time (s)')
        plt.ylabel('State')
        print("BB", end="")

    else:
        print("usage : python <filename> --<parameter : p/pd/bb>")
        exit()

    print(" Control Simulator. Press 'q' to exit.")
    plt.legend()
    plt.grid(True)
    plt.show()