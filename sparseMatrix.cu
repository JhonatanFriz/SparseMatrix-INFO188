#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <ctime>
#include <vector>
#include <tuple>

using namespace std;

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
    float matriz[n][n];
    float multiplicador[n];
    float resultado[n];
    vector<tuple<float,int>> resumen; //resumen es optimo para d <= 0.5
    float limite = n * n * d;
    //cout << "limite" << limite << endl;
    float contador = 0.0;


    for (int i = 0; i < n; i++) {
        float va = (rand() % 100) + 1;
        multiplicador[i] = va;
        for (int j = 0; j < n; ++j) {
            matriz[i][j] = 0.0;
        }
    }

    while (contador<limite){
        float valorAleatorio = (rand() % 100) + 1;
        int usado = rand() % (n * n);
        int fila = usado / n;
        int columna = usado % n;
        if(matriz[fila][columna]==0.0){
            // cout << valorAleatorio << endl;
            matriz[fila][columna] = valorAleatorio; // Asegurar que el valor no sea cero
            resumen.push_back(make_tuple(valorAleatorio,usado));
            contador=contador+1.0;
            // cout << contador << endl;
        }
    }

    // Imprimir la matriz dispersa
    cout << "\nMatriz dispersa generada:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cout << matriz[i][j] << " ";
        }
        cout << "\n";
    }
    cout << "\nVector multiplicador generado:\n";
    for (int i = 0; i < n; ++i) {
        cout << multiplicador[i] << " ";
    }
    cout << "\nVector matriz resumida:\n";
    for (const auto& floatTuple : resumen) {
        cout << "(" << get<0>(floatTuple) << ", " << get<1>(floatTuple) << ")\n";
    }

    return 0;
}