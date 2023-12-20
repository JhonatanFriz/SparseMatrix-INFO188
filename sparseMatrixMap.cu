#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <ctime>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <cmath>

using namespace std;

void calcularCPU(const unordered_map<int, float>& Md, float* m, float* r, int n, int nt) {
    #pragma omp parallel for num_threads(nt) shared(Md, m, r)
    for (auto it = Md.begin(); it != Md.end(); ++it) {
        int fila = it->first / n;
        int columna = it->first % n;
        r[fila] += it->second * m[columna];
    }
}

// void calcularCPU(unordered_map<int, float> Md, float* m, float* r, int n, int nt) {
//     #pragma omp parallel for num_threads(nt) shared(m, r) firstprivate(Md)
//     for (const auto& par : Md) {
//         int fila = (par.first) / n;
//         int columna = (par.first) % n;
//         #pragma omp atomic
//         r[fila] += par.second * m[columna];
//     }
// }

// __global__ void calcularGPU(unordered_map<int,float> Md, float* m, float* r, int n){
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
    
// }

int main(int argc, char* argv[]){
    if (argc != 6) {
        cout << "Error. Debe ejecutarse como ./prog <n> <d> <m> <s> <nt>" << endl;
        exit(EXIT_FAILURE);
    }

    int n = atoi(argv[1]);
    float d = atof(argv[2]);
    bool m = atoi(argv[3]);
    int s = atoi(argv[4]);
    int nt = atoi(argv[5]);

    srand(s);
    float multiplicador[n];
    float resultado[n];
    unordered_map<int, float> Md; //es optimo para d < 0.5
    int limite = round(n*n*d);
    int contador = 0;

    for (int i = 0; i < n; i++) {
        multiplicador[i] = 1.0 + (rand() * 99.0 / (float)RAND_MAX);
        resultado[i] = 0.0;
    }

    while(contador<limite){
        int usado = rand() % (n * n);
        if (Md.find(usado) == Md.end()){
            Md[usado] = 1.0 + (rand() * 99.0 / (float)RAND_MAX);
            contador++;
        }
    }
    
    for (const auto& par : Md) {
        cout << "Posicion: " << par.first << ", Valor: " << par.second << endl;
    }
    cout << "Vector multiplicador generado:\n";
    for (int i = 0; i < n; ++i) {
        cout << multiplicador[i] << " ";
    }

    double tiempoInicial_CPU = omp_get_wtime();
    calcularCPU(Md, multiplicador, resultado, n, nt);
    double tiempoFinal_CPU = omp_get_wtime();
    double tiempoCPU = (tiempoFinal_CPU - tiempoInicial_CPU);
    cout << "\nTiempo CPU: "<< tiempoCPU << " [s]\n";

    cout << "Vector resultado generado:\n";
    for (int i = 0; i < n; ++i) {
        cout << resultado[i] << " ";
    }

    return 0;
}
//probando