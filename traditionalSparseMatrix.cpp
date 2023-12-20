#include <iostream>
#include <vector>
#include <ctime>

// Estructura para representar una entrada no nula en la matriz dispersa
struct EntradaMatrizDispersa {
    int fila;
    int columna;
    double valor;
};

// Función para generar una matriz dispersa aleatoria
void generarMatrizDispersa(int n, double d, int semilla, std::vector<EntradaMatrizDispersa>& matriz) {
    srand(semilla);
    matriz.clear();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if ((double)rand() / RAND_MAX < d) {
                matriz.push_back({i, j, (double)rand() / RAND_MAX});
            }
        }
    }
}

// Función para multiplicar una matriz dispersa por un vector
void multiplicarMatrizPorVector(int n, const std::vector<EntradaMatrizDispersa>& matriz, const std::vector<double>& vector, std::vector<double>& resultado) {
    for (int i = 0; i < n; ++i) {
        resultado[i] = 0.0;
    }

    for (const auto& entrada : matriz) {
        resultado[entrada.fila] += entrada.valor * vector[entrada.columna];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Uso: " << argv[0] << " <n> <d> <s>" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);
    double d = std::atof(argv[2]);
    int semilla = std::atoi(argv[3]);

    // Crear la matriz dispersa y el vector
    std::vector<EntradaMatrizDispersa> matriz;
    std::vector<double> vector(n);
    std::vector<double> resultado(n);

    // Generar la matriz dispersa y el vector
    generarMatrizDispersa(n, d, semilla, matriz);
    for (int i = 0; i < n; ++i) {
        vector[i] = (double)rand() / RAND_MAX;
    }
    size_t tamanoMatrizBytes = sizeof(EntradaMatrizDispersa) * matriz.size();
    std::cout << "Tamaño de la matriz: " << tamanoMatrizBytes/(1024.0 * 1024.0) << " MB" << std::endl;
    // Medir el tiempo de ejecución
    clock_t inicio = clock();

    // Realizar la multiplicación
    multiplicarMatrizPorVector(n, matriz, vector, resultado);

    clock_t fin = clock();
    double tiempo = ((double)(fin - inicio)) / CLOCKS_PER_SEC;

    // Imprimir el tiempo de ejecución
    std::cout << "Tiempo de ejecución: " << tiempo << " segundos" << std::endl;

    return 0;
}
