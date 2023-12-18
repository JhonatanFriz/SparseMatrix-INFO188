#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <ctime>
#include <vector>
#include <tuple>
//#include <algorithm>
#include <unordered_map>
#include <cmath>

using namespace std;

void calcularOpenMP(vector<tuple<float,int>> v, float* m, float* r, int n, int nt){
    #pragma omp parallel for num_threads(nt) shared(v,m,r)
    for (const auto& w : v){
        float valor = get<0>(w);
        int posicion = get<1>(w);
        int fila = posicion / n;
        int columna = posicion % n;
        r[fila] = r[fila] + valor*m[columna];
    }
}

// __global__ void calcularGPU(vector<tuple<float,int>> v, float* m, float* r, int n){
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
    //float matriz[n][n];
    float multiplicador[n];
    float resultado[n];
    unordered_map<int, float> Md;
    vector<tuple<float,int>> resumen; //resumen es optimo para d < 0.5
    float limite = n * n * d;
    //float contador = 0.0;


    for (int i = 0; i < n; i++) {
        multiplicador[i] = 1.0 + (rand() * 99.0 / (float)RAND_MAX);
        resultado[i] = 0.0;
        /*for (int j = 0; j < n; ++j) {
            matriz[i][j] = 0.0;
        }*/
    }

    while (contador<limite){
        float valorAleatorio = 1.0 + (rand() * 99.0 / (float)RAND_MAX);
        int usado = rand() % (n * n);
        /*int fila = usado / n;
        int columna = usado % n;
        if(matriz[fila][columna]==0.0){
            // cout << valorAleatorio << endl;
            matriz[fila][columna] = valorAleatorio; // Asegurar que el valor no sea cero
            resumen.push_back(make_tuple(valorAleatorio,usado));
            contador=contador+1.0;
            // cout << contador << endl;
        }*/
        bool used = false;
        for (const auto& tupla : resumen) {
            int valorEntero = get<1>(tupla);
            if (valorEntero == usado)
                used=true;
        }
        auto iterador = find_if(resumen.begin(),resumen.end(), [=](const tuple<float, int>& tupla) {
        return get<1>(tupla) == usado;
        });
        if (iterador == resumen.end()){
            resumen.push_back(make_tuple(valorAleatorio,usado));
            contador=contador+1.0;
        }
    }
    //TRANSFORMAR VECTOR A UNORDERED MAP
    
    // Imprimir la matriz dispersa
    /*cout << "\nMatriz dispersa generada:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cout << matriz[i][j] << " ";
        }
        cout << "\n";
    }
    cout << "Vector multiplicador generado:\n";
    for (int i = 0; i < n; ++i) {
        cout << multiplicador[i] << " ";
    }
    cout << "\nVector matriz resumida:\n";
    for (const auto& floatTuple : resumen) {
        cout << "(" << get<0>(floatTuple) << ", " << get<1>(floatTuple) << ")\n";
    }*/
    double tiempoInicial_CPU = omp_get_wtime();
    calcularOpenMP(resumen, multiplicador, resultado, n, nt);
    double tiempoFinal_CPU = omp_get_wtime();
    double tiempoCPU = (tiempoFinal_CPU - tiempoInicial_CPU);
    cout << "\nTiempo CPU: "<< tiempoCPU << " [s]\n";
    /*cout << "Vector resultado generado:\n";
    for (int i = 0; i < n; ++i) {
        cout << resultado[i] << " ";
    }*/
    return 0;
}