#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <ctime>
#include <vector>
#include <tuple>
#include <algorithm>
#include <cmath>

using namespace std;

struct casilla {
    int entero;
    float real;
};

void calcularCPU(casilla* v, float* m, float* r, int tv, int n, int nt){
    #pragma omp parallel for num_threads(nt) shared(v,m,r)
    for (int i=0; i<tv; i++){
        int fila = (v[i].entero) / n;
        int columna = (v[i].entero) % n;
        r[fila] = r[fila] + (v[i].real)*m[columna];
    }
}

__global__ void calcularGPU(casilla* v, float* m, float* r, int tv, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < tv){
        int fila = (v[i].entero) / n;
        int columna = (v[i].entero) % n;
        atomicAdd(&r[fila], (v[i].real) * m[columna]);
    }
}

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
    int limite = round(n * n * d);
    casilla matriz[limite];
    int contador = 0;

    for (int i = 0; i < n; i++) {
        multiplicador[i] = 1.0 + (rand() * 99.0 / (float)RAND_MAX);
        resultado[i] = 0.0;
    }

    while (contador < limite){
        int valeat = rand() % (n * n);
        int buscador = 0;
        bool used = false;
        while((buscador<limite)&&(!used)) {
            if (matriz[buscador].entero==valeat)
                used=true;
            buscador++;
        }
        if (!used){
            matriz[contador].entero = valeat;
            matriz[contador].real = 1.0 + (rand() * 99.0 / (float)RAND_MAX);
            contador++;
        }
    }
    
    /*cout << "Vector multiplicador generado:\n";
    for (int i = 0; i < n; ++i) {
        cout << multiplicador[i] << " ";
    }
    cout << endl;
    
    cout << "Vector matriz dispersa:\n";
    for (int i = 0; i< limite; i++){
        cout << "Valor: " << matriz[i].real << " Posicion: " << matriz[i].entero << endl;
    }*/
    
    int total_size = limite * sizeof(casilla);
    cout << "Tamano de la matriz dispersa abreviada: " << total_size/(1024.0 * 1024.0) << " [MB]" << endl;

    if (m==0){
        double tiempoInicial_CPU = omp_get_wtime();
        calcularCPU(matriz, multiplicador, resultado, limite, n, nt);
        double tiempoFinal_CPU = omp_get_wtime();
        double tiempoCPU = (tiempoFinal_CPU - tiempoInicial_CPU);
        cout << "Tiempo CPU: "<< tiempoCPU << " [s]\n";
    }
    else{
        double tiempoInicial_GPU, tiempoFinal_GPU, tiempoGPU;
        float* mul = nullptr;
        float* res = nullptr;
        casilla* mat = nullptr;

        cudaMalloc(&mul, n * sizeof(float));
        cudaMalloc(&res, n * sizeof(float));
        cudaMalloc(&mat, total_size);
        cudaMemcpy(mul, multiplicador, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(res, resultado, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(mat, matriz, total_size, cudaMemcpyHostToDevice);

        int GRID_SIZE = (limite + 127) / 128;
        tiempoInicial_GPU = omp_get_wtime();
        calcularGPU<<<GRID_SIZE, 128>>>(mat, mul, res, limite, n);
        cudaDeviceSynchronize();
        cudaMemcpy(resultado, res, n * sizeof(float), cudaMemcpyDeviceToHost);

        tiempoFinal_GPU = omp_get_wtime();
        cudaFree(mul);
        cudaFree(res);
        cudaFree(mat);
        tiempoGPU = tiempoFinal_GPU - tiempoInicial_GPU;
        cout << "Tiempo GPU: " << tiempoGPU << " [s]\n";
    }

    /*cout << "Vector resultado generado:\n";
    for (int i = 0; i < n; ++i) {
        cout << resultado[i] << " ";
    }*/

    /*delete[] resultado;
    delete[] multiplicador;*/

    return 0;
}