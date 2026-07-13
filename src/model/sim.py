import numpy as np
import matplotlib.pyplot as plt
import sys
from itertools import combinations, chain

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from scipy.optimize import differential_evolution
from functools import partial


def enable_legend_toggles(ax, lines):
    legend = ax.legend()
    legend_lookup = {}

    for legend_item, line in zip(legend.get_lines(), lines):
        legend_item.set_picker(True)
        legend_item.set_pickradius(5)
        legend_lookup[legend_item] = line

    def on_pick(event):
        legend_item = event.artist
        if legend_item not in legend_lookup:
            return

        line = legend_lookup[legend_item]
        visible = not line.get_visible()
        line.set_visible(visible)
        legend_item.set_alpha(1.0 if visible else 0.2)
        ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect('pick_event', on_pick)


def powerset(iterable):
    items = list(iterable)
    return chain.from_iterable(combinations(items, r) for r in range(len(items)+1))


def get_circuit(f, touch_nodes, r_ladder, noise_std, cap_dev):
    circuit = Circuit(f'Circuit @ {f} Hz')

    circuit.SinusoidalVoltageSource(
        'input',
        'vin_clean',
        circuit.gnd,
        amplitude=5@u_V,
        frequency=f@u_kHz
    )

    if noise_std > 0:
        circuit.raw_spice += (
            f"Vrandnoise vin vin_clean TRRANDOM(2 1u 0 {noise_std} 0)\n"
        )
        circuit.R('lpf_r', 'vin', 'vlpf', 2@u_kOhm)
    else:
        circuit.R('lpf_r', 'vin_clean', 'vlpf', 2@u_kOhm)

    circuit.C('lpf_c', 'vlpf', circuit.gnd, 100@u_pF)

    circuit.C('hpf_c', 'vlpf', 'vhpf', 4.7@u_nF)
    circuit.R('hpf_r', 'vhpf', circuit.gnd, 34.2@u_kOhm)

    circuit.VCVS('amp', 'vamp', circuit.gnd, 'vhpf', circuit.gnd, 2)

    circuit.R(1, 'vamp', 'n1', 20@u_kOhm)
    circuit.L("inductor", 'n1', 'd-pad0', 100@u_mH)
    circuit.R("d-padR0", 'd-pad0', 'd-pad1', r_ladder[0])
    circuit.R("d-padR1", 'd-pad1', 'd-pad2', r_ladder[1])
    circuit.R("d-padR2", 'd-pad2', 'd-pad3', r_ladder[2])
    circuit.R("d-padR3", 'd-pad3', 'd-pad4', r_ladder[3])
    circuit.R("r_pad4_gnd", 'd-pad4', circuit.gnd, 135@u_kOhm)
    for touch_node in touch_nodes:
        circuit.C(f"d-padC{touch_node}", f'd-pad{touch_node}', f'finger{touch_node}', 100@u_pF + cap_dev)
        circuit.R(f"d-padRG{touch_node}", f'finger{touch_node}', circuit.gnd, 1.5@u_kOhm)

    circuit.VCVS('buffer', 'vbuf', circuit.gnd, 'n1', circuit.gnd, 1)

    circuit.R('lpf2_r', 'vbuf', 'vlpf2', 2@u_kOhm)
    circuit.C('lpf2_c', 'vlpf2', circuit.gnd, 100@u_pF)

    circuit.D('d1', 'vlpf2', 'venv', model='Dideal')
    circuit.R('env_r', 'venv', circuit.gnd, 10@u_kOhm)
    circuit.C('env_c', 'venv', circuit.gnd, 660@u_nF)

    circuit.R('adc', 'venv', circuit.gnd, 100@u_MOhm)

    circuit.model('Dideal', 'D', IS=1e-15)

    return circuit


def simulate_point(f, touch_nodes, r_ladder, noise_std=0, cap_dev=0@u_pF,
                    step_time=5@u_us, end_time=20@u_ms):
    circuit = get_circuit(f, touch_nodes, r_ladder, noise_std, cap_dev)
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(step_time=step_time, end_time=end_time)
    venv = np.array(analysis['venv'])
    return np.mean(venv[-10:])


def sweep_all_combos(test_cases, frequencies, r_ladder, debug, **sim_kwargs):
    data = np.zeros((len(test_cases), len(frequencies)))
    for i, n in enumerate(test_cases):
        for j, f in enumerate(frequencies):
            data[i, j] = simulate_point(f, n, r_ladder, **sim_kwargs)
            if debug:
                draw_progress(i, len(test_cases), j + 1, len(frequencies), f)
        if debug:
            sys.stdout.write('\n')
            sys.stdout.flush()
    if not debug:
        sys.stdout.write("finished 1 sweep\n")
        sys.stdout.flush()
    return data


def separation_score(data):
    mu, sigma = data.mean(axis=0), data.std(axis=0) + 1e-12
    z = (data - mu) / sigma
 
    n = len(z)
    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(z[i] - z[j])
            min_dist = min(min_dist, d)
    return min_dist


def draw_progress(node_index, n_cases, progress, total, frequency):
    filled = int(30 * progress / total)
    bar = '█' * filled + '-' * (30 - filled)
    sys.stdout.write(
        f'\rCase {node_index}/{n_cases} [{bar}] {progress:4d}/{total} '
        f'({frequency:7.1f} kHz)'
    )
    sys.stdout.flush()


def objective(log_r_ladder, test_cases, opt_frequencies):
    r_ladder = 10 ** np.array(log_r_ladder)
    try:
        data = sweep_all_combos(test_cases, opt_frequencies, r_ladder, False)
    except Exception:
        return 1e6
    return -separation_score(data)


def optimize_ladder(test_cases, opt_frequencies, bounds_ohm,
                     maxiter, popsize, seed_values):
    log_bounds = [(np.log10(bounds_ohm[0]), np.log10(bounds_ohm[1]))] * 4
    n_pop = popsize * 4

    log_seed = np.log10(np.array(seed_values, dtype=float))
    rng = np.random.default_rng(0)
    init_population = log_seed + rng.uniform(-0.5, 0.5, size=(n_pop, 4))
    lo = np.array([b[0] for b in log_bounds])
    hi = np.array([b[1] for b in log_bounds])
    init_population = np.clip(init_population, lo, hi)
    
    result = differential_evolution(
        partial(objective, test_cases=test_cases, opt_frequencies=opt_frequencies),
        log_bounds, maxiter=maxiter, popsize=popsize, tol=1e-3, seed=0,
        workers=-1, updating='deferred', polish=False, init=init_population
    )
    best_r_ladder = 10 ** result.x
    print('\nBest r_ladder (ohms):', best_r_ladder)
    print('Separation score:', -result.fun)
    return best_r_ladder


if __name__ == '__main__':
    nodes = 4
    test_cases = list(powerset(range(1, nodes + 1)))

    opt_frequencies = np.linspace(1, 250, 50)

    #best_r_ladder = optimize_ladder(test_cases, opt_frequencies,
    #                                 bounds_ohm=(1e3, 200e3),
    #                                 maxiter=20, popsize=8,
    #                                 seed_values=[10e3, 15e3, 30e3, 60e3])

    best_r_ladder = [4.558e3, 5.264e3, 15.762e3, 48.299e3]
    #best_r_ladder = [10e3, 15e3, 30e3, 60e3]
    frequencies = np.arange(1, 350, 2)
 
    data = sweep_all_combos(test_cases, frequencies, best_r_ladder, True, noise_std=0.02, cap_dev=0@u_pF)
 
    np.savetxt('data.txt', data, fmt='%.6f')
    np.savetxt('best_r_ladder.txt', best_r_ladder, fmt='%.1f')
 
    plt.figure()
    lines = []
    for n in range(len(test_cases)):
        line, = plt.plot(frequencies, data[n], label=f'Touching Node {test_cases[n]}')
        lines.append(line)
    plt.title(f'Output Voltage vs Frequency (r_ladder={np.round(best_r_ladder, 1)})')
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('Output Voltage [V]')
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', alpha=0.5)
    plt.grid(which='minor', linestyle=':', alpha=0.3)
    enable_legend_toggles(plt.gca(), lines)
    plt.show()
    plt.close()
