#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <ctime>
#include <vector>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <cmath>

using namespace std;

struct casilla {
    int entero;
    float real;
};

void calcularCPU(casilla* v, float* m, float* r, int tv, int n, int nt){//(vector<tuple<float,int>> v, float* m, float* r, int n, int nt){
    #pragma omp parallel for num_threads(nt) shared(v,m,r)
    for (int i=0; i<tv; i++){
        //float valor = get<0>(w);
        //int posicion = get<1>(w);
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
        //printf("Agregando: %d\n", (v[i].real)*m[columna]);
        // r[fila] = r[fila] + (v[i].real)*m[columna];
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
    //double tiempoA, tiempoB, tiempoC;
    //tiempoA = omp_get_wtime();
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
    /*tiempoB = omp_get_wtime();
    tiempoC = (tiempoB - tiempoA);
    cout << "Tiempo de formacion de matriz: "<< tiempoC << " [s]\n";*/
    
    /*cout << "Vector multiplicador generado:\n";
    for (int i = 0; i < n; ++i) {
        cout << multiplicador[i] << " ";
    }
    cout << endl;
    
    cout << "Vector matriz dispersa:\n";
    for (int i = 0; i< limite; i++){
        cout << "Valor: " << matriz[i].real << " Posicion: " << matriz[i].entero << endl;
    }*/
    
    if (m==0){
        double tiempoInicial_CPU = omp_get_wtime();
        calcularCPU(matriz, multiplicador, resultado, limite, n, nt);
        double tiempoFinal_CPU = omp_get_wtime();
        double tiempoCPU = (tiempoFinal_CPU - tiempoInicial_CPU);
        cout << "Tiempo CPU: "<< tiempoCPU << " [s]\n";
    }
    else{
        double tiempoInicial_GPU, tiempoFinal_GPU, tiempoGPU;
        tiempoInicial_GPU = omp_get_wtime();
        float* mul = nullptr;
        float* res = nullptr;
        casilla* mat = nullptr;
        int total_size = limite * sizeof(casilla);

        cudaMalloc(&mul, n * sizeof(float));
        cudaMalloc(&res, n * sizeof(float));
        cudaMalloc(&mat, total_size);
        cudaMemcpy(mul, multiplicador, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(res, resultado, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(mat, matriz, total_size, cudaMemcpyHostToDevice);

        int GRID_SIZE = (limite + 127) / 128;
        calcularGPU<<<GRID_SIZE, 128>>>(mat, mul, res, limite, n);
        cudaDeviceSynchronize();
        cudaMemcpy(resultado, res, n * sizeof(float), cudaMemcpyDeviceToHost);

        tiempoFinal_GPU = omp_get_wtime();
        cudaFree(mul);
        cudaFree(res);
        cudaFree(mat);
        tiempoGPU = tiempoFinal_GPU - tiempoInicial_GPU;
        cout << "Tiempo GPU: " << tiempoGPU << " [s]\n";
        // double tiempoInicial_GPU, tiempoFinal_GPU, tiempoGPU;
        // tiempoInicial_GPU = omp_get_wtime();
        // int* mul = 0;
        // int* res = 0;
        // casilla* mat = 0;
        // int total_size = (limite) * sizeof(casilla);
        // cudaMalloc(&mul,sizeof(multiplicador));
        // cudaMalloc(&res,sizeof(resultado));
        // cudaMalloc(&mat,total_size);
        // cudaMemcpy(mul,multiplicador,sizeof(multiplicador),cudaMemcpyHostToDevice);
        // cudaMemcpy(res,resultado,sizeof(resultado),cudaMemcpyHostToDevice);
        // cudaMemcpy(mat,matriz,total_size,cudaMemcpyHostToDevice);
        // dim3 GRID_SIZE = ((limite+127)/128);
        // dim3 BLOCK_SIZE = (128);
        // calcularGPU<<<GRID_SIZE,BLOCK_SIZE>>>(matriz, multiplicador, resultado, limite, n);
        // cudaMemcpy(res,resultado,sizeof(resultado),cudaMemcpyDeviceToHost);
        // cudaDeviceSynchronize();
        // tiempoFinal_GPU= omp_get_wtime();
        // tiempoGPU = tiempoFinal_GPU - tiempoInicial_GPU;
        // cout << "Tiempo GPU: "<< tiempoGPU << " [s]\n";
        // cudaFree(mul);
        // cudaFree(res);
        // cudaFree(mat);
    }

    /*cout << "Vector resultado generado:\n";
    for (int i = 0; i < n; ++i) {
        cout << resultado[i] << " ";
    }*/

    /*delete[] resultado;
    delete[] multiplicador;*/

    return 0;
}