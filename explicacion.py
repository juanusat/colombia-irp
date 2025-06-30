import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# === CONFIGURACIÓN SEGÚN base_modelo_ceplex.txt ===
client_names = [
    'Depot Armenia', 'Barranq 01', 'Barranq 02', 'Bogota', 'Cali 01',
    'Cali 02', 'Cali 03', 'Cali 04', 'Cali 05'
]
name_to_index = {name: i for i, name in enumerate(client_names)}
depot_idx = name_to_index['Depot Armenia']
n_total_nodos = len(client_names)  # 10 nodos (0=depósito, 1..9=clientes)
num_clientes = n_total_nodos - 1  # 9 clientes

original_num_vehiculos_reales = 3
original_capacidad_vehiculo_unit = 1500
original_demanda_clientes_original = [
    0,    # Depósito
    200, 200, 600, 100, 300, 700, 700, 200, 500
]
original_demanda_clientes_AI = {i: original_demanda_clientes_original[i] for i in range(1, num_clientes + 1)}
original_cantidad_total_producto_deposito = 12000
original_costo_penalizacion_desabastecimiento_unit = 100

original_costo_inventario_unitario_por_cliente = {
    1: 6,  2: 4,  3: 5,  4: 7,  5: 6,
    6: 3,  7: 6,  8: 7,  9: 5
}

INFINITE_COST = 999999999

distancia_matrix_data_completa = [
  [0, 1098, 1098, 286, 194, 194, 194, 194, 194, 935],
  [1098, 0, 1302, 1212, 1212, 1212, 1212, 1212, 926, 1302],
  [1098, 1302, 0, 1212, 1212, 1212, 1212, 1212, 926, 1302],
  [286, 1212, 1212, 0, 484, 484, 484, 484, 484, 649],
  [194, 1212, 1212, 484, 0, 4, 4, 4, 4, 1133],
  [194, 1212, 1212, 484, 4, 0, 4, 4, 4, 1133],
  [194, 1212, 1212, 484, 4, 4, 0, 4, 4, 1133],
  [194, 1212, 1212, 484, 4, 4, 4, 0, 4, 1133],
  [194, 926, 926, 484, 4, 4, 4, 4, 0, 1133],
  [935, 1302, 1302, 649, 1133, 1133, 1133, 1133, 1133, 0]
]

original_costos_base_ij = {}
for i in range(n_total_nodos):
    for j in range(n_total_nodos):
        original_costos_base_ij[(i, j)] = INFINITE_COST

for r_idx in range(len(distancia_matrix_data_completa)):
    for c_idx in range(len(distancia_matrix_data_completa[r_idx])):
        row_node = r_idx
        col_node = c_idx
        if row_node < n_total_nodos and col_node < n_total_nodos:
            original_costos_base_ij[(row_node, col_node)] = distancia_matrix_data_completa[r_idx][c_idx]

for i in range(n_total_nodos):
    for j in range(n_total_nodos):
        if i == j:
            original_costos_base_ij[(i, j)] = 0
        elif original_costos_base_ij.get((i, j), INFINITE_COST) == INFINITE_COST and \
                original_costos_base_ij.get((j, i), INFINITE_COST) != INFINITE_COST:
            original_costos_base_ij[(i, j)] = original_costos_base_ij[(j, i)]
        elif original_costos_base_ij.get((j, i), INFINITE_COST) == INFINITE_COST and \
                original_costos_base_ij.get((i, j), INFINITE_COST) != INFINITE_COST:
            original_costos_base_ij[(j, i)] = original_costos_base_ij[(i, j)]


def phase1_inventory_assignment(available_inventory, client_demands, costo_penalizacion_unit, costo_inventario_unitario_por_cliente):
    print("\n--- FASE I: Problema de Asignación de Inventarios (AI) ---")
    print(f"    Inventario total disponible en depósito (A): {available_inventory}")

    assigned_quantities = {i: 0 for i in client_demands.keys()}  # Inicializa cantidades asignadas a 0 para cada cliente
    unsatisfied_quantities = {i: 0 for i in client_demands.keys()}  # Inicializa cantidades no satisfechas a 0
    total_unsatisfied_demand_cost = 0  # Acumula el costo total de penalización
    total_inventory_holding_cost = 0  # Acumula el costo total de inventario
    current_inventory = available_inventory  # Inventario disponible en el depósito
    clients_to_be_routed = []  # Lista de clientes a rutear

    sorted_client_indices = sorted(client_demands.keys())  # Ordena los índices de clientes

    for client_idx in sorted_client_indices:
        demand = client_demands[client_idx]  # Demanda del cliente actual
        if current_inventory > 0:
            quantity_to_assign = min(current_inventory, demand)  # Asigna lo que se pueda según inventario
            assigned_quantities[client_idx] = quantity_to_assign  # Registra la cantidad asignada
            current_inventory -= quantity_to_assign  # Actualiza el inventario restante
            
            costo_unitario = costo_inventario_unitario_por_cliente.get(client_idx, 0)  # Costo unitario de inventario
            costo_por_cliente_entregado = quantity_to_assign * costo_unitario  # Costo total de inventario para el cliente
            total_inventory_holding_cost += costo_por_cliente_entregado  # Suma al total de inventario
            
            print(f"    Cliente {client_names[client_idx]} (Demanda {demand}): Se asignan {quantity_to_assign} unidades (Costo Inv. {costo_unitario}/u). Inventario restante: {current_inventory}")
            if quantity_to_assign > 0:
                clients_to_be_routed.append(client_idx)  # Agrega cliente a la lista de ruteo si recibió algo
        else:
            print(f"    Cliente {client_names[client_idx]} (Demanda {demand}): No hay inventario disponible para asignar.")

    for client_idx in sorted_client_indices:
        unsatisfied_quantities[client_idx] = client_demands[client_idx] - assigned_quantities[client_idx]  # Calcula demanda no satisfecha
        if unsatisfied_quantities[client_idx] > 0:
            cost_for_client = unsatisfied_quantities[client_idx] * costo_penalizacion_unit  # Calcula penalización
            total_unsatisfied_demand_cost += cost_for_client  # Suma al total de penalización
            print(f"    Cliente {client_names[client_idx]}: {unsatisfied_quantities[client_idx]} unidades no satisfechas. Costo penalización: {cost_for_client}")

    print(f"\n    Costo Total de Penalización por Demanda No Satisfecha: {total_unsatisfied_demand_cost}")
    print(f"    Costo Total de Inventario por Unidades Entregadas: {total_inventory_holding_cost}")

    return assigned_quantities, unsatisfied_quantities, total_unsatisfied_demand_cost, total_inventory_holding_cost, clients_to_be_routed

def calculate_savings(depot_idx, clients, distance_matrix):
    savings = {}  # Diccionario para almacenar los ahorros
    for i in clients:
        for j in clients:
            if i != j:
                s_ij = distance_matrix.get((depot_idx, i), INFINITE_COST) + \
                             distance_matrix.get((j, depot_idx), INFINITE_COST) - \
                             distance_matrix.get((i, j), INFINITE_COST)  # Fórmula de ahorros Clarke & Wright
                if s_ij != INFINITE_COST:
                    savings[(i, j)] = s_ij  # Guarda el ahorro para el par (i, j)
    return sorted(savings.items(), key=lambda item: item[1], reverse=True)  # Ordena de mayor a menor ahorro

def clarke_wright_savings_parallel(depot_idx, clients_to_route, demands, vehicle_capacity_unit, num_vehiculos_disponibles, distance_matrix):
    print("\n--- FASE II (Paso 1): Algoritmo de Ahorros de Clarke & Wright (Paralelo) ---")

    if not clients_to_route:
        print("    No hay clientes para rutear en esta fase.")
        return [], []

    client_to_route_map = {}  # Mapea cada cliente a su ruta actual
    current_routes = []  # Lista de rutas actuales
    current_route_loads = []  # Cargas actuales de cada ruta
    
    route_endpoints = {}  # Guarda los extremos de cada ruta

    for i in clients_to_route:
        new_route_idx = len(current_routes)  # Índice de la nueva ruta
        current_routes.append([depot_idx, i, depot_idx])  # Inicializa ruta: depósito-cliente-depósito
        current_route_loads.append(demands[i])  # Carga inicial es la demanda del cliente
        client_to_route_map[i] = new_route_idx  # Asocia cliente a la ruta
        route_endpoints[new_route_idx] = {'start': i, 'end': i}  # Extremos de la ruta
        
    print(f"    Inicialización: {len(clients_to_route)} rutas individuales creadas.")

    savings = calculate_savings(depot_idx, clients_to_route, distance_matrix)  # Calcula ahorros posibles
    print(f"    Calculados {len(savings)} posibles ahorros.")

    for (i, j), saving_value in savings:
        if i not in client_to_route_map or j not in client_to_route_map:
            continue  # Si alguno ya fue fusionado, salta

        route_i_idx = client_to_route_map[i]  # Ruta de i
        route_j_idx = client_to_route_map[j]  # Ruta de j

        if route_i_idx == route_j_idx:
            continue  # No fusionar la misma ruta
            
        endpoints_i = route_endpoints.get(route_i_idx)  # Extremos de la ruta de i
        endpoints_j = route_endpoints.get(route_j_idx)  # Extremos de la ruta de j

        if endpoints_i is None or endpoints_j is None:
            continue  # Si alguna ruta ya fue fusionada, salta

        can_merge = False  # Bandera para saber si se puede fusionar
        new_route_nodes = []  # Nueva ruta fusionada
        new_route_load = 0  # Nueva carga
        new_start_client = -1  # Nuevo inicio de la ruta
        new_end_client = -1  # Nuevo fin de la ruta

        if i == endpoints_i['end'] and j == endpoints_j['start']:
            temp_route_i_nodes = current_routes[route_i_idx]  # Ruta de i
            temp_route_j_nodes = current_routes[route_j_idx]  # Ruta de j
            
            new_route_nodes = temp_route_i_nodes[:-1] + temp_route_j_nodes[1:]  # Fusiona rutas
            new_route_load = current_route_loads[route_i_idx] + current_route_loads[route_j_idx]  # Suma cargas
            
            new_start_client = endpoints_i['start']  # Nuevo inicio
            new_end_client = endpoints_j['end']  # Nuevo fin
            can_merge = True  # Se puede fusionar

        elif j == endpoints_j['end'] and i == endpoints_i['start']:
            temp_route_j_nodes = current_routes[route_j_idx]  # Ruta de j
            temp_route_i_nodes = current_routes[route_i_idx]  # Ruta de i
            
            new_route_nodes = temp_route_j_nodes[:-1] + temp_route_i_nodes[1:]  # Fusiona rutas
            new_route_load = current_route_loads[route_j_idx] + current_route_loads[route_i_idx]  # Suma cargas
            
            new_start_client = endpoints_j['start']  # Nuevo inicio
            new_end_client = endpoints_i['end']  # Nuevo fin
            can_merge = True  # Se puede fusionar

        if can_merge and new_route_load <= vehicle_capacity_unit:
            new_route_idx = len(current_routes)  # Índice de la nueva ruta
            current_routes.append(new_route_nodes)  # Agrega la nueva ruta
            current_route_loads.append(new_route_load)  # Agrega la nueva carga
            route_endpoints[new_route_idx] = {'start': new_start_client, 'end': new_end_client}  # Actualiza extremos

            for client_node in current_routes[route_i_idx][1:-1]:
                client_to_route_map[client_node] = new_route_idx  # Actualiza mapeo de clientes
            for client_node in current_routes[route_j_idx][1:-1]:
                client_to_route_map[client_node] = new_route_idx  # Actualiza mapeo de clientes

            current_routes[route_i_idx] = None  # Marca rutas fusionadas como None
            current_routes[route_j_idx] = None  # Marca rutas fusionadas como None
            current_route_loads[route_i_idx] = 0  # Carga a cero para rutas fusionadas
            current_route_loads[route_j_idx] = 0  # Carga a cero para rutas fusionadas
            del route_endpoints[route_i_idx]  # Elimina extremos de la ruta fusionada
            del route_endpoints[route_j_idx]  # Elimina extremos de la ruta fusionada

    final_routes = []  # Rutas finales
    final_route_loads = []  # Cargas finales
    
    vehicle_counter = 0
    for route_idx, route in enumerate(current_routes):
        if route is not None and len(route) > 2:
            if vehicle_counter < num_vehiculos_disponibles:
                final_routes.append(route)  # Agrega ruta si hay vehículos disponibles
                final_route_loads.append(current_route_loads[route_idx])
                vehicle_counter += 1
            else:
                print(f"    Advertencia: Algoritmo C&W generó más rutas factibles que vehículos disponibles. Clientes en la ruta {route} no serán ruteados por falta de vehículo.")
    
    print(f"    C&W: Se consolidaron las rutas, resultando en {len(final_routes)} rutas para {num_vehiculos_disponibles} vehículos disponibles.")
    return final_routes, final_route_loads

def solve_tsp_for_route(route_id, route_nodes_with_depot, original_client_indices_in_route, distance_matrix):
    if not original_client_indices_in_route:
        return 0, [route_nodes_with_depot[0], route_nodes_with_depot[0]]  # Si no hay clientes, ruta trivial

    tsp_nodes_original_indices = [depot_idx] + sorted(original_client_indices_in_route)  # Nodos para el TSP
    tsp_node_to_original_index = {i: original_idx for i, original_idx in enumerate(tsp_nodes_original_indices)}  # Mapeo local->original
    original_to_tsp_node_index = {original_idx: i for i, original_idx in enumerate(tsp_nodes_original_indices)}  # Mapeo original->local

    num_tsp_nodes = len(tsp_nodes_original_indices)  # Número de nodos en el TSP
    
    tsp_sub_distance_matrix = {}  # Matriz de distancias para el subproblema TSP
    for i_local in range(num_tsp_nodes):
        for j_local in range(num_tsp_nodes):
            original_i = tsp_node_to_original_index[i_local]
            original_j = tsp_node_to_original_index[j_local]
            tsp_sub_distance_matrix[(i_local, j_local)] = distance_matrix.get((original_i, original_j), INFINITE_COST)  # Distancia entre nodos
            if i_local == j_local:
                tsp_sub_distance_matrix[(i_local, j_local)] = 0  # Costo cero para la diagonal

    manager = pywrapcp.RoutingIndexManager(num_tsp_nodes, 1, original_to_tsp_node_index[depot_idx])  # Crea el gestor de índices
    routing = pywrapcp.RoutingModel(manager)  # Crea el modelo de ruteo

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return tsp_sub_distance_matrix.get((from_node, to_node), INFINITE_COST)  # Devuelve la distancia entre nodos

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)  # Registra la función de distancia
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)  # Asigna la función de costo

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()  # Parámetros de búsqueda
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)  # Estrategia inicial
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)  # Metaheurística
    search_parameters.time_limit.seconds = 30  # Límite de tiempo

    assignment = routing.SolveWithParameters(search_parameters)  # Resuelve el TSP

    route_cost = 0  # Costo de la ruta
    optimized_route_nodes = []  # Ruta optimizada

    if assignment:
        index = routing.Start(0)
        while not routing.IsEnd(index):
            previous_index = index
            original_node = tsp_node_to_original_index[manager.IndexToNode(index)]  # Nodo original
            optimized_route_nodes.append(original_node)  # Agrega nodo a la ruta
            index = assignment.Value(routing.NextVar(index))  # Siguiente nodo
            route_cost += routing.GetArcCostForVehicle(previous_index, index, 0)  # Suma el costo
        original_node = tsp_node_to_original_index[manager.IndexToNode(index)]  # Nodo final
        optimized_route_nodes.append(original_node)
        
        return route_cost, optimized_route_nodes  # Devuelve costo y ruta
    else:
        print(f"    Advertencia: No se encontró una ruta TSP para el Vehículo {route_id} con clientes {original_client_indices_in_route}.")
        return INFINITE_COST, []  # Si no hay solución, retorna costo infinito

def solve_inventory_routing_problem_three_phase(
    current_cantidad_total_producto_deposito,
    current_capacidad_vehiculo_unit,
    current_costo_penalizacion_desabastecimiento_unit,
    current_demanda_clientes_AI,
    current_num_vehiculos_reales,
    current_costos_base_ij,
    current_costo_inventario_unitario_por_cliente
):
    assigned_quantities, unsatisfied_quantities, total_unsatisfied_demand_cost, total_inventory_holding_cost, clients_to_be_routed = \
        phase1_inventory_assignment(
            current_cantidad_total_producto_deposito,
            current_demanda_clientes_AI,
            current_costo_penalizacion_desabastecimiento_unit,
            current_costo_inventario_unitario_por_cliente
        )

    print("\n--- FASE II (Paso 1): Ejecutando Clarke & Wright Savings Algorithm ---")
    demands_for_routing = {c_idx: assigned_quantities[c_idx] for c_idx in clients_to_be_routed if assigned_quantities[c_idx] > 0}
    grouped_routes_cws, grouped_route_loads_cws = clarke_wright_savings_parallel(
        depot_idx=depot_idx,
        clients_to_route=clients_to_be_routed,
        demands=demands_for_routing,
        vehicle_capacity_unit=current_capacidad_vehiculo_unit,
        num_vehiculos_disponibles=current_num_vehiculos_reales,
        distance_matrix=current_costos_base_ij
    )
    total_transport_cost = 0
    routes_details = {}

    print("\n--- FASE III: Resolución de TSP para cada ruta definida por C&W ---")
    
    for k_route_idx, cws_route in enumerate(grouped_routes_cws):
        if k_route_idx >= current_num_vehiculos_reales:
            print(f"    Saltando vehículo {k_route_idx + 1}: Excede el número de vehículos disponibles ({current_num_vehiculos_reales}).")
            continue

        vehicle_id_for_display = k_route_idx + 1
        
        clients_in_this_route_original_indices = cws_route[1:-1]
        
        if clients_in_this_route_original_indices:
            print(f"    Resolviendo TSP para Vehículo {vehicle_id_for_display} con clientes: {[client_names[c] for c in clients_in_this_route_original_indices]}")
            
            cost_of_this_route, optimized_route_nodes_path = solve_tsp_for_route(
                vehicle_id_for_display,
                cws_route,
                clients_in_this_route_original_indices,
                current_costos_base_ij
            )
            
            total_transport_cost += cost_of_this_route
            routes_details[vehicle_id_for_display] = {
                'cost': cost_of_this_route,
                'path': [client_names[node_idx] for node_idx in optimized_route_nodes_path],
                'clients_visited': [client_names[c] for c in clients_in_this_route_original_indices],
                'load': grouped_route_loads_cws[k_route_idx]
            }
        else:
            print(f"    Vehículo {vehicle_id_for_display} (ruta C&W {k_route_idx}) no tiene clientes reales asignados. Ruta: {[client_names[node] for node in cws_route]}.")

    total_objective_value = total_transport_cost + total_unsatisfied_demand_cost + total_inventory_holding_cost

    print("\n--- Resumen Final del Procedimiento Goloso (3 Fases) ---")
    print(f"Costo Total de la Solución Golosa: {total_objective_value}")
    print(f"    Costo Total de Transporte: {total_transport_cost}")
    print(f"    Costo Total por Demanda No Satisfecha (de Fase I): {total_unsatisfied_demand_cost}")
    print(f"    Costo Total de Inventario por Unidades Entregadas: {total_inventory_holding_cost}")

    print("\n--- Cantidades Entregadas (w_i) y No Satisfechas (u_i) (resultados de Fase I) ---")
    total_delivered = 0
    total_original_demand = sum(current_demanda_clientes_AI.values())
    
    for i in range(1, num_clientes + 1):
        delivered = assigned_quantities.get(i, 0)
        unsatisfied = unsatisfied_quantities.get(i, 0)
        print(f"    {client_names[i]} (Demanda original: {current_demanda_clientes_AI.get(i,0)}): {delivered} unidades entregadas, {unsatisfied} no satisfechas.")
        total_delivered += delivered
    print(f"    Total demandada (todos los clientes): {total_original_demand}")
    print(f"    Total entregada: {total_delivered}")
    print(f"    Inventario inicial en depósito: {current_cantidad_total_producto_deposito}")
    print(f"    Inventario restante después de Fase I: {current_cantidad_total_producto_deposito - total_delivered}")

    print("\n--- Rutas, Carga y Costos por Vehículo (resultantes de Fases II y III) ---")
    if not routes_details:
        print("    No se generaron rutas de vehículos reales.")
    else:
        for v_display_id in sorted(routes_details.keys()):
            detail = routes_details[v_display_id]
            print(f"    Vehículo {v_display_id}:")
            print(f"Ruta: {' -> '.join(detail['path'])}")
            print(f"    Clientes visitados: {', '.join(detail['clients_visited'])}")
            print(f"    Carga de la ruta: {detail['load']} / {current_capacidad_vehiculo_unit}")
            print(f"    Costo de Ruta: {detail['cost']}")
            print("-" * 30)
            
    # Calcular costo promedio por ruta
    costo_promedio_por_ruta = total_transport_cost / len(routes_details) if len(routes_details) > 0 else 0
    
    return {
        'penalizacion_unit': current_costo_penalizacion_desabastecimiento_unit,
        'capacidad_vehiculo': current_capacidad_vehiculo_unit,
        'num_vehiculos': current_num_vehiculos_reales,
        'costo_total_solucion': total_objective_value,
        'costo_transporte': total_transport_cost,
        'costo_penalizacion_desabastecimiento': total_unsatisfied_demand_cost,
        'costo_inventario_entregado': total_inventory_holding_cost,
        'num_rutas_generadas': len(routes_details),
        'total_demanda_original': total_original_demand,
        'total_entregado': total_delivered,
        'inventario_final_deposito': current_cantidad_total_producto_deposito - total_delivered,
        'costo_promedio_por_ruta': costo_promedio_por_ruta
    }

if __name__ == '__main__':
    penalizacion_desabastecimiento_rangos = [100, 50, 80, 120]  # Diferentes valores de penalización por desabastecimiento
    capacidad_vehiculo_rangos = [1500, 1200, 800, 500]  # Diferentes capacidades de los vehículos
    
    costo_inventario_fijo_para_escenarios = original_costo_inventario_unitario_por_cliente  # Costos de inventario fijos

    todos_los_resultados_sensibilidad = []  # Lista para almacenar resultados de todos los escenarios

    output_folder = "informe_sensibilidad_graficas"  # Carpeta para guardar las gráficas
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Crea la carpeta si no existe
        print(f"\nCarpeta '{output_folder}' creada para guardar las gráficas.")

    for penalizacion_val in penalizacion_desabastecimiento_rangos:  # Itera sobre los valores de penalización
        for capacidad_val in capacidad_vehiculo_rangos:  # Itera sobre las capacidades de los vehículos
                print(f"\n{'='*60}")
                print(f"--- INICIANDO NUEVO ESCENARIO: Penalización={penalizacion_val}, Capacidad={capacidad_val} ---")
                print(f"{'='*60}\n")
                
                # Ejecuta el modelo para el escenario actual
                resultado_escenario = solve_inventory_routing_problem_three_phase(
                    current_cantidad_total_producto_deposito=original_cantidad_total_producto_deposito,  # Inventario inicial
                    current_capacidad_vehiculo_unit=capacidad_val,  # Capacidad del vehículo
                    current_costo_penalizacion_desabastecimiento_unit=penalizacion_val,  # Penalización por desabastecimiento
                    current_demanda_clientes_AI=original_demanda_clientes_AI,  # Demanda de los clientes
                    current_num_vehiculos_reales=original_num_vehiculos_reales,  # Número de vehículos disponibles
                    current_costos_base_ij=original_costos_base_ij,  # Matriz de costos de viaje
                    current_costo_inventario_unitario_por_cliente=costo_inventario_fijo_para_escenarios  # Costos de inventario
                )
                
                todos_los_resultados_sensibilidad.append(resultado_escenario)  # Almacena el resultado del escenario
                print(f"\n--- FIN DEL ESCENARIO: Penalización={penalizacion_val}, Capacidad={capacidad_val} ---\n")

    print("\n\n" + "="*80)
    print("--- INFORME DE SENSIBILIDAD COMPLETO DE TODOS LOS ESCENARIOS ---")
    print("="*80)
    df_resultados = pd.DataFrame(todos_los_resultados_sensibilidad)  # Convierte los resultados a un DataFrame
    print(df_resultados.to_string())  # Muestra los resultados en formato tabular

    # Nuevo bloque para imprimir la FO de cada escenario al final
    print("\n--- Valores de la Función Objetivo (FO) para cada escenario ---")
    for idx, row in df_resultados.iterrows():
        print(f"Escenario P={row['penalizacion_unit']}, C={row['capacidad_vehiculo']}: FO = {row['costo_total_solucion']:.2f}")  # Imprime la FO

    plt.figure(figsize=(15, len(df_resultados) * 0.3 + 2))  # Configura el tamaño de la figura
    ax = plt.subplot(111, frame_on=False)  # Crea un subplot sin marco
    ax.xaxis.set_visible(False)  # Oculta el eje X
    ax.yaxis.set_visible(False)  # Oculta el eje Y
    table = pd.plotting.table(ax, df_resultados, loc='center', cellLoc='center')  # Crea una tabla con los resultados
    table.auto_set_font_size(False)
    table.set_fontsize(9)  # Ajusta el tamaño de la fuente
    table.scale(2.4, 1.2)  # Escala la tabla
    plt.title('Informe de Sensibilidad - Resumen Tabular', fontsize=14)  # Título de la tabla
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'informe_sensibilidad_tabla_resumen.png'))  # Guarda la tabla como imagen
    plt.show()  # Muestra la tabla

    print("\n--- Generando y guardando Gráficos de Sensibilidad ---")
    try:
        
        plt.figure(figsize=(12, 7))
        sns.lineplot(
            data=df_resultados,
            x='penalizacion_unit',
            y='costo_total_solucion',
            hue='capacidad_vehiculo',
            marker='o',
            palette='viridis'
        )
        plt.title('Costo Total de la Solución vs. Costo de Penalización', fontsize=14)
        plt.xlabel('Costo de Penalización por Desabastecimiento Unitario', fontsize=12)
        plt.ylabel('Costo Total de la Solución', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Capacidad Vehículo', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'costo_total_vs_penalizacion.png'))
        plt.show()

        plt.figure(figsize=(12, 7))
        sns.lineplot(
            data=df_resultados,
            x='capacidad_vehiculo',
            y='costo_total_solucion',
            hue='penalizacion_unit',
            marker='o',
            palette='magma'
        )
        plt.title('Costo Total de la Solución vs. Capacidad del Vehículo', fontsize=14)
        plt.xlabel('Capacidad de Vehículo Unitario', fontsize=12)
        plt.ylabel('Costo Total de la Solución', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Costo Penalización', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'costo_total_vs_capacidad.png'))
        plt.show()

        plt.figure(figsize=(12, 7))
        sns.lineplot(
            data=df_resultados,
            x='capacidad_vehiculo',
            y='num_rutas_generadas',
            hue='penalizacion_unit',
            marker='o',
            palette='cividis'
        )
        plt.title('Número de Rutas Generadas vs. Capacidad del Vehículo', fontsize=14)
        plt.xlabel('Capacidad de Vehículo Unitario', fontsize=12)
        plt.ylabel('Número de Rutas Generadas', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Costo Penalización', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'num_rutas_vs_capacidad.png'))
        plt.show()

        plt.figure(figsize=(12, 7))
        sns.lineplot(
            data=df_resultados,
            x='penalizacion_unit',
            y='total_entregado',
            hue='capacidad_vehiculo',
            marker='o',
            palette='plasma'
        )
        plt.title('Total de Producto Entregado vs. Costo de Penalización', fontsize=14)
        plt.xlabel('Costo de Penalización por Desabastecimiento Unitario', fontsize=12)
        plt.ylabel('Total de Producto Entregado', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Capacidad Vehículo', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'total_entregado_vs_penalizacion.png'))
        plt.show()

        plt.figure(figsize=(12, 7))
        sns.lineplot(
            data=df_resultados,
            x='capacidad_vehiculo',
            y='costo_transporte',
            hue='penalizacion_unit',
            marker='o',
            palette='coolwarm'
        )
        plt.title('Costo de Transporte vs. Capacidad del Vehículo', fontsize=14)
        plt.xlabel('Capacidad de Vehículo Unitario', fontsize=12)
        plt.ylabel('Costo de Transporte', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Costo Penalización', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'costo_transporte_vs_capacidad.png'))
        plt.show()

        plt.figure(figsize=(12, 7))
        sns.lineplot(
            data=df_resultados,
            x='penalizacion_unit',
            y='costo_inventario_entregado',
            hue='capacidad_vehiculo',
            marker='o',
            palette='rocket'
        )
        plt.title('Costo de Inventario Entregado vs. Costo de Penalización', fontsize=14)
        plt.xlabel('Costo de Penalización por Desabastecimiento Unitario', fontsize=12)
        plt.ylabel('Costo de Inventario Entregado', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Capacidad Vehículo', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'costo_inventario_entregado_vs_penalizacion.png'))
        plt.show()

        # Nueva gráfica: Costo Promedio por Ruta vs. Capacidad del Vehículo
        plt.figure(figsize=(12, 7))
        sns.lineplot(
            data=df_resultados,
            x='capacidad_vehiculo',
            y='costo_promedio_por_ruta',
            hue='penalizacion_unit',
            marker='o',
            palette='viridis',
            linewidth=2.5,
            markersize=8
        )
        plt.title('Costo Promedio por Camión/Ruta vs. Capacidad del Vehículo', fontsize=16, fontweight='bold')
        plt.xlabel('Capacidad del Vehículo (unidades)', fontsize=12)
        plt.ylabel('Costo Promedio por Ruta', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Costo de Penalización', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Agregar anotaciones para mostrar valores específicos
        for penalizacion in df_resultados['penalizacion_unit'].unique():
            subset = df_resultados[df_resultados['penalizacion_unit'] == penalizacion]
            for _, row in subset.iterrows():
                plt.annotate(f'{row["costo_promedio_por_ruta"]:.0f}', 
                           (row['capacidad_vehiculo'], row['costo_promedio_por_ruta']),
                           textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'costo_promedio_por_ruta_vs_capacidad.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Gráfica adicional: Gráfica de barras agrupadas para mejor visualización
        plt.figure(figsize=(14, 8))
        
        # Crear gráfico de barras agrupadas
        capacidades = df_resultados['capacidad_vehiculo'].unique()
        penalizaciones = df_resultados['penalizacion_unit'].unique()
        
        x = range(len(capacidades))
        width = 0.2
        
        for i, penalizacion in enumerate(penalizaciones):
            subset = df_resultados[df_resultados['penalizacion_unit'] == penalizacion]
            costos = [subset[subset['capacidad_vehiculo'] == cap]['costo_promedio_por_ruta'].iloc[0] for cap in capacidades]
            
            plt.bar([xi + width*i for xi in x], costos, width, 
                   label=f'Penalización = {penalizacion}', alpha=0.8)
            
            # Agregar valores en las barras
            for j, costo in enumerate(costos):
                plt.text(x[j] + width*i, costo + 5, f'{costo:.0f}', 
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.xlabel('Capacidad del Vehículo (unidades)', fontsize=12)
        plt.ylabel('Costo Promedio por Ruta', fontsize=12)
        plt.title('Costo Promedio por Camión/Ruta según Capacidad del Vehículo\n(Comparación por Diferentes Costos de Penalización)', 
                 fontsize=14, fontweight='bold')
        plt.xticks([xi + width*1.5 for xi in x], capacidades)
        plt.legend(title='Costo de Penalización', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'costo_promedio_por_ruta_barras_agrupadas.png'), dpi=300, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"\nOcurrió un error al generar o guardar las gráficas: {e}")
        print("\nPara la visualización de resultados, asegúrate de instalar las librerías `matplotlib`, `seaborn` y `pandas`.")
        print("Puedes instalarlas con: `pip install matplotlib seaborn pandas`")
        
print("\n" + "="*80)
print("--- ANÁLISIS DE SENSIBILIDAD COMPLETADO ---")
print(f"Las gráficas se han guardado en la carpeta: '{output_folder}'")
print('=' * 80)