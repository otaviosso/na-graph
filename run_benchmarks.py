import subprocess
import argparse
import os
import sys

# Mapeamento de benchmarks para o valor do -j no GraphOne
GRAPHONE_BENCHMARK_MAP = {
    "bfs": 0,
    "pr": 1,
    "cc_sv": 2,
    "bc": 3,
}

def validate_arguments(framework, benchmark, args):
    """Valida se todos os argumentos necessários estão presentes."""
    required_args = {
        "dgap": ["base_file", "dynamic_file", "output_path", "source_node", "trials"],
        "BAL": ["base_file", "dynamic_file", "output_path", "source_node", "trials"],
        "CSR": ["base_file", "dynamic_file", "output_path", "source_node", "trials"],
        "GraphOne": ["base_file", "dynamic_file", "output_path", "source_node", "vertices", "threads"],
        "XPGraph": ["base_file", "dynamic_file", "output_path", "source_node", "vertices", "threads", "recovery_dir", "query_times"],
    }
    for arg in required_args[framework]:
        value = getattr(args, arg)
        if value is None:
            print(f"Erro: O argumento --{arg} é obrigatório para o framework {framework}.")
            sys.exit(1)
        
        # Verificações adicionais para tipos de dados
        if arg in ["source_node", "trials", "vertices", "threads", "query_times"]:
            if not isinstance(value, int):
                print(f"Erro: O argumento --{arg} deve ser um número inteiro.")
                sys.exit(1)

def run_benchmark(framework, benchmark, base_file, dynamic_file, output_path, source_node, trials, vertices, threads, recovery_dir, query_times):
    # Define o diretório do framework
    framework_dir = os.path.join(framework)
    
    # Define o comando base com base no framework e benchmark
    if framework in ["dgap", "BAL", "CSR"]:
        command = [
            os.path.join(framework_dir, f"./{benchmark}"),  # Executável dentro da pasta do framework
            "-B", base_file, "-D", dynamic_file, "-f", output_path, "-r", str(source_node), "-n", str(trials), "-a"
        ]
    elif framework == "GraphOne":
        # Mapeia o benchmark para o valor do -j
        job = GRAPHONE_BENCHMARK_MAP.get(benchmark)
        if job is None:
            raise ValueError(f"Benchmark {benchmark} não suportado para o GraphOne.")
        command = [
            os.path.join(framework_dir, "./graphone"),  # Executável dentro da pasta do framework
            "-b", base_file, "-d", dynamic_file, "-o", output_path, "-v", str(vertices), "-j", str(job), "-s", str(source_node), "-t", str(threads)
        ]
    elif framework == "XPGraph":
        # Updated command for XPGraph to run separate executables (e.g., ./pr, ./bfs, etc.)
        executable = os.path.join(framework_dir, f"./{benchmark}")
        command = [
            executable,
            "-f", base_file, "-p0", output_path, "--recovery", recovery_dir, "--source", str(source_node),
            "-v", str(vertices), "-q", str(query_times), "-t", str(threads)
        ]
    else:
        raise ValueError(f"Framework {framework} não suportado.")
    
    # Executa o comando
    print(f"Executando {benchmark} no {framework}...")
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Exibe a saída
    print(f"Saída do {framework} ({benchmark}):")
    print(result.stdout)
    if result.stderr:
        print(f"Erro do {framework} ({benchmark}):")
        print(result.stderr)

def main():
    parser = argparse.ArgumentParser(description="Executa benchmarks para frameworks de grafos.")
    parser.add_argument("framework", choices=["dgap", "GraphOne", "BAL", "XPGraph", "CSR"], help="Framework de grafos a ser utilizado.")
    parser.add_argument("benchmark", choices=["pr", "cc_sv", "bfs", "bc"], help="Benchmark a ser executado.")
    parser.add_argument("--base_file", help="Arquivo de base do grafo.")
    parser.add_argument("--dynamic_file", help="Arquivo dinâmico do grafo.")
    parser.add_argument("--output_path", help="Caminho para armazenar o grafo.")
    parser.add_argument("--source_node", type=int, help="Nó de origem, no dgap 'source vertex-id'.")
    parser.add_argument("--trials", type=int, default=1, help="Número de tentativas.")
    parser.add_argument("--vertices", type=int, help="Número de vértices (apenas para GraphOne e XPGraph).")
    parser.add_argument("--threads", type=int, default=1, help="Número de threads (apenas para GraphOne e XPGraph).")
    parser.add_argument("--recovery_dir", help="Diretório de recuperação (apenas para XPGraph).")
    parser.add_argument("--query_times", type=int, help="Número de vezes para executar a análise (apenas para XPGraph).")
    args = parser.parse_args()
    
    # Valida os argumentos necessários
    validate_arguments(args.framework, args.benchmark, args)
    
    run_benchmark(
        framework=args.framework,
        benchmark=args.benchmark,
        base_file=args.base_file,
        dynamic_file=args.dynamic_file,
        output_path=args.output_path,
        source_node=args.source_node,
        trials=args.trials,
        vertices=args.vertices,
        threads=args.threads,
        recovery_dir=args.recovery_dir,
        query_times=args.query_times
    )

main()