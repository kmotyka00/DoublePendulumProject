import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
import numpy as np

class Simulator:
    def __init__(self, model, solver, cart_width=0.15, cart_height=0.1, skip_frames=40):
        self.model = model
        self.solver = solver
        self.solution = None
        self.cart_width = cart_width
        self.cart_height = cart_height
        self.skip_frames = skip_frames

    def render_animation(self, solution=None):
        if solution is None:
            solution = self.solution

        if solution is None:
            raise Exception('No solution to render. Call solve() first or use simulate() method.')

        x_solution = solution[:, 0]
        a_solution = solution[:, 2]
        b_solution = solution[:, 4]

        skip_frames = self.skip_frames

        x_solution = x_solution[::skip_frames]
        a_solution = a_solution[::skip_frames]
        b_solution = b_solution[::skip_frames]

        cart_width = self.cart_width
        cart_height = self.cart_height

        dt = self.solver.dt

        frames = len(x_solution)

        j1_x = self.model.l1 * np.sin(a_solution) + x_solution
        j1_y = self.model.l1 * np.cos(a_solution)

        j2_x = self.model.l2 * np.sin(b_solution) + j1_x
        j2_y = self.model.l2 * np.cos(b_solution) + j1_y

        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1, 1), ylim=(-1, 1))
        ax.set_aspect('equal')
        ax.grid()

        patch = ax.add_patch(Rectangle((0, 0), 0, 0, linewidth=1, edgecolor='k', facecolor='r'))

        line, = ax.plot([], [], 'o-', lw=2)
        time_template = 'time: %.1f s'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        def init():
            line.set_data([], [])
            time_text.set_text('')
            patch.set_xy((-cart_width / 2, -cart_height / 2))
            patch.set_width(cart_width)
            patch.set_height(cart_height)
            return line, time_text


        def animate(i):
            thisx = [x_solution[i], j1_x[i], j2_x[i]]
            thisy = [0, j1_y[i], j2_y[i]]

            line.set_data(thisx, thisy)
            now = i * skip_frames * dt
            time_text.set_text(time_template % now)

            patch.set_x(x_solution[i] - cart_width / 2)
            return line, time_text, patch


        output_animation = animation.FuncAnimation(fig, animate, frames=frames,
                                    interval=1, blit=True, init_func=init)
        
        plt.close(fig)
        output_animation.save('controlled_dipc.gif', writer='imagemagick', fps=24)
        return output_animation
    
    def render_plot(self, solution=None):
        if solution is None:
            solution = self.solution

        if solution is None:
            raise Exception('No solution to render. Call solve() first or use simulate() method.')
        
        x_solution = solution[:, 0]
        a_solution = solution[:, 2]
        b_solution = solution[:, 4]

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(self.solver.times, a_solution[:len(self.solver.times)] % 2*np.pi, label='a')
        plt.title('a')
        plt.subplot(1, 3, 2)
        plt.plot(self.solver.times, b_solution[:len(self.solver.times)] % 2*np.pi, label='b')
        plt.title('b')
        plt.subplot(1, 3, 3)
        plt.plot(self.solver.times, x_solution[:len(self.solver.times)], label='x')
        plt.title('x')
        plt.show()
    
    def simulate(self):
        self.solution = self.solver.solve(self.model)
        self.render_plot()
        return self.render_animation()
    