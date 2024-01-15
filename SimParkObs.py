import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.optimize import minimize
import time


def CarSim(options, ModelPredictiveControl):
    ###settings##################################################

    # start = time.clock()

    FIG_SIZE = options["FIG_SIZE"]
    obstacle = options["OBSTACLES"]

    mpc = ModelPredictiveControl(
        options["PARKING_POSITION"], options["OBSTACLE_POSITION"]
    )

    no_of_inputs = 2  # two inputs for acceleration and steering wheel angle

    u = np.zeros(mpc.horizon * no_of_inputs)

    bounds = []

    for i in range(mpc.horizon):
        bounds += [[-1, 1]]
        bounds += [[-0.8, 0.8]]

    ref_1 = mpc.reference1
    ref_2 = mpc.reference2

    ref = ref_1

    state_i = np.array([[0, 0, 0, 0]])  # list of states the car will be in
    u_i = np.array([[0, 0]])  # list on inputs the car will actually execute

    sim_total = 250  # number of times the mpc will make preictions
    predicted_info = [
        state_i
    ]  # this 3D array will have the values of states in all the 250 iterations

    # ACTUAL CODE##############################

    for i in range(1, sim_total + 1):
        start_time = time.time()
        # optimization
        u_solution = minimize(
            mpc.cost_function,
            u,
            (state_i[-1], ref),
            method="SLSQP",
            bounds=bounds,
            tol=1e-5,
        )

        u = u_solution.x

        print(
            "Step "
            + str(i)
            + " of "
            + str(sim_total)
            + "   Time "
            + str(round(time.time() - start_time, 5))
        )

        y = mpc.plant_model(
            state_i[-1], mpc.dt, u[0], u[1]
        )  # y= output of the system(x,y,velocity,angle) NOTE:Only one input value is executed out of the 20 predicted ones
        predicted_states = np.array([y])
        for k in range(1, mpc.horizon):
            predictions = mpc.plant_model(
                predicted_states[-1], mpc.dt, u[2 * k], u[2 * k + 1]
            )  # predcitons will have row matrix of preiction of the very next state
            predicted_states = np.append(
                predicted_states, np.array([predictions]), 0
            )  # this row matrix will be added as a row in predicted_state matrix which will have 20 rows
            # when it gets out of the loop
        predicted_info += [
            predicted_states
        ]  # predicted_info will have the 250 matrices of predicted states when oit gets out of the loop
        state_i = np.append(
            state_i, np.array([y]), 0
        )  # the state when we getny executing the first predicted input is now stored as a row in matrix state_i
        u_i = np.append(
            u_i, np.array([(u[0], u[1])]), 0
        )  # the one exectuted input value is stored in u_i

    #########Display#################

    fig = plt.figure(figsize=(FIG_SIZE[0], FIG_SIZE[1]))  # defining the output window

    gs = gridspec.GridSpec(8, 8)  #

    ax = fig.add_subplot(gs[:8, :8])  # creating a plot named 'ax' and of size (8,8)

    plt.xlim(-3, 17)  # car  initial postition will be 0,0. SO taking
    plt.ylim(-3, 17)  # 0,0 away from the meeting point of two axes

    plt.xticks(np.arange(0, 11, step=2))  # range of number on X and Y axes
    plt.yticks(np.arange(0, 11, step=2))

    plt.title("MPC 2D")

    car_width = 1.0
    patch_car = mpatches.Rectangle(
        (0, 0), car_width, 2.5, fc="k", fill=False
    )  # creating a rectangle to repesent a car
    patch_goal = mpatches.Rectangle(
        (0, 0), car_width, 2.5, fc="b", ls="dashdot", fill=False
    )  # reacting a rectangle to represemt the parking space

    ax.add_patch(patch_car)  # add car to the plot
    ax.add_patch(patch_goal)  # add parking space to the plot

    (predict,) = ax.plot(
        [], [], "r--", linewidth=1
    )  # visual representation of pedicted path

    if obstacle:
        patch_obs = mpatches.Circle(
            (mpc.x_obs, mpc.y_obs), 0.5
        )  # add a circular obstacle in the pat
        ax.add_patch(patch_obs)

    # this function updates the positon of the car
    def car_patch_pos(x, y, psi):  # The origin of the car is on the top left hand
        # return [x,y]                           #side.The origin is set to zero. SO we have to
        x_new = (
            x - np.sin(psi) * car_width / 2
        )  # shift the car a little upward to aline it
        y_new = y + np.cos(psi) * car_width / 2  # with the prediction plot
        return [x_new, y_new]

    def update_plot(num):
        patch_car.set_xy(
            car_patch_pos(state_i[num, 0], state_i[num, 1], state_i[num, 2])
        )
        patch_car.angle = np.rad2deg(state_i[num, 2]) - 90

        if num <= 130 or ref_2 == None:
            patch_goal.set_xy(car_patch_pos(ref_1[0], ref_1[1], ref_1[2]))
            patch_goal.angle = np.rad2deg(ref_1[2]) - 90

        else:
            patch_goal.set_xy(car_patch_pos(ref_1[0], ref_1[1], ref_1[2]))
            patch_goal.angle = np.rad2deg(ref_2[2]) - 90

        predict.set_data(
            predicted_info[num][:, 0], predicted_info[num][:, 1]
        )  # visual representation of predicted path

        return patch_car

    animation.FuncAnimation(
        fig,
        update_plot,
        frames=range(1, len(state_i)),
        interval=100,
        repeat=True,
        blit=False,
    )

    plt.show()
