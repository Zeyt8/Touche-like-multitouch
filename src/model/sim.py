import numpy as np
import matplotlib.pyplot as plt
import sys

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

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

    circuit.VCVS('amp', 'vamp', circuit.gnd, 'vhpf', circuit.gnd, 2)

    circuit.R(1, 'vamp', 'n1', 220@u_kOhm)
    circuit.L("inductor", 'n1', 'd-pad1', 100@u_mH)
    circuit.R("d-padR1", 'd-pad1', 'd-pad2', 150@u_kOhm)
    circuit.R("d-padR2", 'd-pad2', 'd-pad3', 150@u_kOhm)
    circuit.R("d-padR3", 'd-pad3', 'd-pad4', 150@u_kOhm)
    circuit.R("short", 'd-pad4', circuit.gnd, 1@u_Ohm)
    for touch_node in touch_nodes:
        circuit.C("d-padC", f'd-pad{touch_node}', circuit.gnd, 100@u_pF)

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

    plt.plot(t, vin, label='Vin')
    plt.plot(t, vlpf, label='LPF')
    plt.plot(t, vhpf, label='HPF')
    plt.plot(t, vamp, label='Amplifier')
    plt.plot(t, n1, label='N1')
    plt.plot(t, vbuf, label='Buffer')
    plt.plot(t, vlpf2, label='LPF2')
    plt.plot(t, venv, label='Envelope')

    plt.title(f'Full Signal Chain @ {f:.1f} kHz, Touching Node {n}')
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [V]')
    plt.grid(True)
    plt.legend()

    plt.show()

def draw_progress(node_index, progress, total, frequency):
    filled = int(bar_width * progress / total)
    bar = '█' * filled + '-' * (bar_width - filled)
    sys.stdout.write(
        f'\rNode {node_index}/{nodes} [{bar}] {progress:4d}/{total} '
        f'({frequency:7.1f} kHz)'
    )
    sys.stdout.flush()

if __name__ == '__main__':
    nodes = 4
    freq_step = 17.5
    frequencies = np.arange(1, 3500 + freq_step, freq_step)
    data = np.zeros((nodes+1, len(frequencies)))

    bar_width = 30

    for n in range(0, nodes + 1):
        for i, f in enumerate(frequencies):
            circuit = get_circuit([n] if n > 0 else [])

            simulator = circuit.simulator(temperature=25, nominal_temperature=25)

            analysis = simulator.transient(
                step_time=10@u_us,
                end_time=5@u_ms
            )

            #plot_all(analysis, f, n)
            venv = np.array(analysis['venv'])
            data[n, i] = np.mean(venv[-10:])

            draw_progress(n, i + 1, len(frequencies), f)

        sys.stdout.write('\n')
        sys.stdout.flush()

    np.savetxt('data.txt', data, fmt='%.6f')

    plt.figure()
    for n in range(1, nodes + 1):
        plt.plot(frequencies, data[n-1], label=f'Touching Node {n}')
    plt.title('Output Voltage vs Frequency')
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('Output Voltage [V]')
    plt.grid(True)
    plt.legend()
    plt.show()
