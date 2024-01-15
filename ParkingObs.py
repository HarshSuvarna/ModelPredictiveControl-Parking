import numpy as np
from SimParkObs import CarSim


options = {}
options["FIG_SIZE"] = [8, 8]
options["OBSTACLES"] = True
options["OBSTACLE_POSITION"] = [6, 6]
options["PARKING_POSITION"] = [
    10,  # X co-ordinate
    10,  # Y co-ordinate
    0,  # Angle (radians)
]


class ModelPredictiveControl:
    def __init__(self, parking_position, obstacle_position):
        self.horizon = 10
        self.dt = 0.2

        self.reference1 = parking_position
        self.reference2 = None  # [10, 10, 90]

        # Position of obstacle
        self.x_obs = obstacle_position[0]
        self.y_obs = obstacle_position[1]

    def plant_model(self, prev_state, dt, pedal, steering):
        # these values can be used to get the next values of X,Y,V & psi at time t+1

        x_t = prev_state[0]  # pevious value of X
        y_t = prev_state[1]  # previous value of Y
        psi_t = prev_state[2]  # previous angle of the car
        v_t = prev_state[3]  # previous value of velocity
        a_t = pedal  # previous value of acceleration
        beta = steering  # previous value of steering

        # formula to calculate the state of the car at time t+1

        x_t_1 = x_t + (v_t * np.cos(psi_t)) * dt
        y_t_1 = y_t + (v_t * np.sin(psi_t)) * dt
        v_t_1 = v_t + a_t * dt - v_t / 25.0
        psi_t_1 = psi_t + (v_t * (np.tan(beta) / 2.5)) * dt

        return [x_t_1, y_t_1, psi_t_1, v_t_1]

    def cost_function(self, u, *args):
        state = args[0]
        ref = args[1]
        cost = 0.0

        # The optimizer will put out multiple sets of input array with different
        # combination of input values. We put all set of the input values one
        # by one in plant model and check the cost of that
        # set of inputs provide by the optimizer

        for k in range(0, self.horizon):
            # punish the system if the car is far way from the goal

            state = self.plant_model(state, self.dt, u[k * 2], u[(k * 2) + 1])
            cost += (state[0] - ref[0]) ** 2  # for X
            cost += (state[1] - ref[1]) ** 2  # for Y

            cost += (state[2] - ref[2]) ** 2  # for angle

            cost += self.obstacle_cost(state[0], state[1])  # for obstacle

        return cost

    def obstacle_cost(self, x, y):
        x1 = self.x_obs
        y1 = self.y_obs
        distance = (
            (x - x1) ** 2 + (y - y1) ** 2
        ) ** 0.5  # calculating the distance betweeen obstacle an
        # and car

        if distance > 2:  # if distance is less than 2 cost will be 50
            return 15
        else:
            return (1 / distance) * 30  # if car goes too close, add more cost


CarSim(options, ModelPredictiveControl)
