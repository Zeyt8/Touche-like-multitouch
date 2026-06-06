import numpy as np
import matplotlib.pyplot as plt
import sys
from itertools import combinations, chain

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *


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

def get_circuit(touch_nodes):
    circuit = Circuit(f'RL + LPF + Envelope @ {f} Hz')

    circuit.SinusoidalVoltageSource(
        'input',
        'vin',
        circuit.gnd,
        amplitude=5@u_V,
        frequency=f@u_kHz
    )

    #circuit.R('r1', 'vin', 'vhpf', 1@u_Ohm)
    circuit.R('lpf_r', 'vin', 'vlpf', 200@u_Ohm)
    circuit.C('lpf_c', 'vlpf', circuit.gnd, 100@u_pF)

    circuit.C('hpf_c', 'vlpf', 'vhpf', 4.7@u_nF)
    circuit.R('hpf_r', 'vhpf', circuit.gnd, 34.2@u_kOhm)

    circuit.VCVS('amp', 'vamp', circuit.gnd, 'vhpf', circuit.gnd, 1.5)

    circuit.R(1, 'vamp', 'n1', 1000@u_kOhm)
    circuit.L("inductor", 'n1', 'd-pad0', 100@u_mH)
    circuit.R("d-padR0", 'd-pad0', 'd-pad1', 800@u_kOhm)
    circuit.R("d-padR1", 'd-pad1', 'd-pad2', 300@u_kOhm)
    circuit.R("d-padR2", 'd-pad2', 'd-pad3', 200@u_kOhm)
    circuit.R("d-padR3", 'd-pad3', 'd-pad4', 100@u_kOhm)
    circuit.R("short", 'd-pad4', circuit.gnd, 100@u_MOhm)
    for touch_node in touch_nodes:
        circuit.C(f"d-padC{touch_node}", f'd-pad{touch_node}', circuit.gnd, 100@u_pF)

    circuit.VCVS('buffer', 'vbuf', circuit.gnd, 'n1', circuit.gnd, 1)

    #circuit.R('lpf2_r', 'vbuf', 'vlpf2', 1@u_Ohm)
    circuit.R('lpf2_r', 'vbuf', 'vlpf2', 200@u_Ohm)
    circuit.C('lpf2_c', 'vlpf2', circuit.gnd, 100@u_pF)


    #circuit.R('env_r', 'vlpf2', 'venv', 1@u_Ohm)
    circuit.D('d1', 'vlpf2', 'venv', model='Dideal')
    circuit.R('env_r', 'venv', circuit.gnd, 10@u_kOhm)
    circuit.C('env_c', 'venv', circuit.gnd, 660@u_nF)

    circuit.R('adc', 'venv', circuit.gnd, 100@u_MOhm)

    circuit.model('Dideal', 'D', IS=1e-15)

    return circuit

def plot_all(analysis, f, n):
    t = np.array(analysis.time)
    vin = np.array(analysis['vin'])
    vlpf = np.array(analysis['vlpf'])
    vhpf = np.array(analysis['vhpf'])
    vamp = np.array(analysis['vamp'])
    n1 = np.array(analysis['n1'])
    vbuf = np.array(analysis['vbuf'])
    vlpf2 = np.array(analysis['vlpf2'])
    venv = np.array(analysis['venv'])

    plt.figure()

    lines = [
        plt.plot(t, vin, label='Vin')[0],
        plt.plot(t, vlpf, label='LPF')[0],
        plt.plot(t, vhpf, label='HPF')[0],
        plt.plot(t, vamp, label='Amplifier')[0],
        plt.plot(t, n1, label='N1')[0],
        plt.plot(t, vbuf, label='Buffer')[0],
        plt.plot(t, vlpf2, label='LPF2')[0],
        plt.plot(t, venv, label='Envelope')[0],
    ]

    plt.title(f'Full Signal Chain @ {f:.1f} kHz, Touching Node {n}')
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [V]')
    plt.grid(True)
    enable_legend_toggles(plt.gca(), lines)

    plt.show()

def draw_progress(node_index, progress, total, frequency):
    filled = int(bar_width * progress / total)
    bar = '█' * filled + '-' * (bar_width - filled)
    sys.stdout.write(
        f'\rCase {node_index}/{len(test_cases)} [{bar}] {progress:4d}/{total} '
        f'({frequency:7.1f} kHz)'
    )
    sys.stdout.flush()

def powerset(iterable):
    items = list(iterable)
    return chain.from_iterable(combinations(items, r) for r in range(len(items)+1))

if __name__ == '__main__':
    nodes = 4
    test_cases = list(powerset(range(1, nodes + 1)))
    #test_cases = [(), (1,), (2,), (3,), (4,)]
    freq_step = 17.5 * 10
    frequencies = np.arange(1, 3500, freq_step)
    data = np.zeros((len(test_cases), len(frequencies)))

    bar_width = 30

    for i, n in enumerate(test_cases):
        for j, f in enumerate(frequencies):
            circuit = get_circuit(n)

            simulator = circuit.simulator(temperature=25, nominal_temperature=25)

            analysis = simulator.transient(
                step_time=10@u_us,
                end_time=10@u_ms
            )

            #plot_all(analysis, f, n)
            venv = np.array(analysis['venv'])
            data[i, j] = np.mean(venv[-10:])

            draw_progress(i, j + 1, len(frequencies), f)

        sys.stdout.write('\n')
        sys.stdout.flush()

    np.savetxt('data.txt', data, fmt='%.6f')

    plt.figure()
    lines = []
    for n in range(len(test_cases)):
        line, = plt.plot(frequencies, data[n], label=f'Touching Node {test_cases[n]}')
        lines.append(line)
    plt.title('Output Voltage vs Frequency')
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('Output Voltage [V]')
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', alpha=0.5)
    plt.grid(which='minor', linestyle=':', alpha=0.3)
    enable_legend_toggles(plt.gca(), lines)
    plt.show()
