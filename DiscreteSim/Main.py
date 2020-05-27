import pandas as pd

from Entry import Entry
from Queue import Queue
from Station import Station
from Exit import Exit
from MachineStation import MachineStation
from PaletStation import PaletStation

from Distributions import fixed, gamma, exponential
from Controller import Controller
from Data import calculate_machine_probs, calculate_waste_probs

def main(simulation):
    controller = Controller(168)

    machine_transitions = calculate_machine_probs()
    waste_probs = calculate_waste_probs()

    entry = Entry("Recibidor", exponential, [5.108333], gamma, [0.913, 2380])
    reception = Queue("Recepcion")
    machine = MachineStation("Maquina", fixed, [0], machine_transitions)

    machine_waste = Exit("Descarte maquina")
    band_12 = Queue("Banda 12")
    band_14 = Queue("Banda 14")
    band_16 = Queue("Banda 16")
    band_18 = Queue("Banda 18")
    band_20 = Queue("Banda 20")
    band_22 = Queue("Banda 22")
    band_24 = Queue("Banda 24")
    band_26 = Queue("Banda 26")
    band_28 = Queue("Banda 28")
    band_30 = Queue("Banda 30")
    band_32 = Queue("Banda 32")

    cali_12 = Station("Calibre 12", fixed, [0.002318], [waste_probs[12], 1-waste_probs[12]])
    cali_14 = Station("Calibre 14", fixed, [0.002318], [waste_probs[14], 1-waste_probs[14]])
    cali_16 = Station("Calibre 16", fixed, [0.002318], [waste_probs[16], 1-waste_probs[16]])
    cali_18 = Station("Calibre 18", fixed, [0.002318], [waste_probs[18], 1-waste_probs[18]])
    cali_20 = Station("Calibre 20", fixed, [0.002031], [waste_probs[20], 1-waste_probs[20]])
    cali_22 = Station("Calibre 22", fixed, [0.000924], [waste_probs[22], 1-waste_probs[22]])
    cali_24 = Station("Calibre 24", fixed, [0.004010], [waste_probs[24], 1-waste_probs[24]])
    cali_26 = Station("Calibre 26", fixed, [0.001590], [waste_probs[26], 1-waste_probs[26]])
    cali_28 = Station("Calibre 28", fixed, [0.002643], [waste_probs[28], 1-waste_probs[28]])
    cali_30 = Station("Calibre 30", fixed, [0.006250], [waste_probs[30], 1-waste_probs[30]])
    cali_32 = Station("Calibre 32", fixed, [0.016444], [waste_probs[32], 1-waste_probs[32]])

    descarte_12 = Exit("Descarte 12")
    descarte_14 = Exit("Descarte 14")
    descarte_16 = Exit("Descarte 16")
    descarte_18 = Exit("Descarte 18")
    descarte_20 = Exit("Descarte 20")
    descarte_22 = Exit("Descarte 22")
    descarte_24 = Exit("Descarte 24")
    descarte_26 = Exit("Descarte 26")
    descarte_28 = Exit("Descarte 28")
    descarte_30 = Exit("Descarte 30")
    descarte_32 = Exit("Descarte 32")

    wait_small_p = Queue("Espera palet pequeño")
    wait_big_p = Queue("Espera palet grande")

    palet_peque = PaletStation("Palet pequeño", fixed, [0], 980)
    palet_grande = PaletStation("Palet grande", fixed, [0], 1180)

    cold_room = Queue("Cuarto frio")
    dispacher = Station("Despachador", exponential, [92.52], [1])
    exit = Exit("Salida")

    entry.connect_with(reception)
    reception.connect_with(machine)

    machine.connect_with(band_12)
    machine.connect_with(band_14)
    machine.connect_with(band_16)
    machine.connect_with(band_18)
    machine.connect_with(band_20)
    machine.connect_with(band_22)
    machine.connect_with(band_24)
    machine.connect_with(band_26)
    machine.connect_with(band_28)
    machine.connect_with(band_30)
    machine.connect_with(band_32)
    machine.connect_with(machine_waste)

    band_12.connect_with(cali_12)
    band_14.connect_with(cali_14)
    band_16.connect_with(cali_16)
    band_18.connect_with(cali_18)
    band_20.connect_with(cali_20)
    band_22.connect_with(cali_22)
    band_24.connect_with(cali_24)
    band_26.connect_with(cali_26)
    band_28.connect_with(cali_28)
    band_30.connect_with(cali_30)
    band_32.connect_with(cali_32)

    cali_12.connect_with(descarte_12) 
    cali_14.connect_with(descarte_14) 
    cali_16.connect_with(descarte_16) 
    cali_18.connect_with(descarte_18) 
    cali_20.connect_with(descarte_20) 
    cali_22.connect_with(descarte_22) 
    cali_24.connect_with(descarte_24) 
    cali_26.connect_with(descarte_26) 
    cali_28.connect_with(descarte_28) 
    cali_30.connect_with(descarte_30) 
    cali_32.connect_with(descarte_32) 

    cali_12.connect_with(wait_big_p) 
    cali_14.connect_with(wait_big_p) 
    cali_16.connect_with(wait_big_p) 
    cali_18.connect_with(wait_big_p) 
    cali_20.connect_with(wait_big_p) 
    cali_22.connect_with(wait_big_p) 
    cali_24.connect_with(wait_small_p) 
    cali_26.connect_with(wait_small_p) 
    cali_28.connect_with(wait_small_p) 
    cali_30.connect_with(wait_small_p) 
    cali_32.connect_with(wait_small_p) 

    wait_small_p.connect_with(palet_peque)
    wait_big_p.connect_with(palet_grande)

    palet_peque.connect_with(cold_room)
    palet_grande.connect_with(cold_room)

    cold_room.connect_with(dispacher)
    dispacher.connect_with(exit)

    entry.execute(None)
    controller.start_simulation()

    print("simulation %d finished" % simulation)

    queues = [band_12, band_14, band_16, band_18, band_20, band_22, band_24, band_26, band_28, band_30, band_32]

    exits = [machine_waste, descarte_12, descarte_14, descarte_16, descarte_18, descarte_20, descarte_22, descarte_24,
             descarte_26, descarte_28, descarte_30, descarte_32, exit]

    stations = [palet_peque, palet_grande, cali_12, cali_14, cali_16, cali_18, cali_20, cali_22, cali_24, cali_26, cali_28, cali_30, cali_32]


    data_queue = []
    for queue in queues:
        name, items, lenght, max_wait, min_wait, avg_wait = queue.get_info()
        data_queue.append([simulation, name, max_wait, avg_wait])

    data_exit = []
    for exit in exits:
        name, kg = exit.get_info()
        data_exit.append([simulation, name, kg])

    data_station = []
    for station in stations:
        name, kg = station.get_info()
        data_station.append([simulation, name, kg])

    return data_queue, data_exit, data_station

data_queue = []
data_exit = []
data_station = []

for simulation in range(100):
    dat_queue, dat_exit, dat_station = main(simulation)
    data_queue += dat_queue
    data_exit += dat_exit
    data_station += dat_station

pd.DataFrame(data_queue, columns=["simulation", "name_queue", "max wait", "avg wait"]).to_pickle("results/queue_result.pkl")
pd.DataFrame(data_exit, columns=["simulation", "name_exit", "kg"]).to_pickle("results/exit_result.pkl")
pd.DataFrame(data_station, columns=["simulation", "name_station", "kg"]).to_pickle("results/station_result.pkl")
