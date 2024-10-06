#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <xmmintrin.h>
#define _GNU_SOURCE

// Função naive para normalizar um vetor de características
/* void normalize_feature_vector(float* features, int length) {
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        sum += features[i] * features[i];
    }
    float inv_sqrt = 1.0f / sqrt(sum);

    for (int i = 0; i < length; i++) {
        features[i] *= inv_sqrt;
    }
} */

// Função otimizada para normalizar um vetor de características usando SSE
void normalize_feature_vector_sse(float* features, int length) {
    __m128 sum_vector = _mm_setzero_ps(); // Inicializa um vetor SSE com zeros
    int i;

    // Calcular a soma dos quadrados
    for (i = 0; i <= length - 4; i += 4) {
        __m128 feature_vector = _mm_loadu_ps(&features[i]); // Carrega 4 elementos do vetor
        __m128 squared_vector = _mm_mul_ps(feature_vector, feature_vector); // Calcula o quadrado
        sum_vector = _mm_add_ps(sum_vector, squared_vector); // Soma
    }

    // Reduzindo a soma dos quadrados para um único valor
    float sum_array[4];
    _mm_storeu_ps(sum_array, sum_vector);
    float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

    // Tratar os elementos restantes se length não for múltiplo de 4
    for (; i < length; i++) {
        sum += features[i] * features[i];
    }

    float inv_sqrt = 1.0f / sqrtf(sum);

    // Normaliza o vetor
    for (i = 0; i < length; i++) {
        features[i] *= inv_sqrt;
    }
}

// Definição da lookup table
float inv_sqrt_table[TABLE_SIZE];

// Função para pré-computar a tabela
void incializaLookupTable() {
    for(int i = 0; i < TABLE_SIZE; i++) {
        inv_sqrt_table[i] = 1.00f / sqrt((float)i);
    }
}

// Função para normalizar o vetor pelo cenário de otimização 1 (Lookup Table)
void normalize_feature_vector(float* features, int length) {
    float sum = 1.00f;

    for(int i = 0; i < length; i++) {
        sum += features[i] * features[i];
    }
    
    if(sum == 0.00f) {
        printf("Não é possível normalizar vetor nulo!");
        return;
    }

    int index = (int)sum;
    if(index >= TABLE_SIZE) {
        index = TABLE_SIZE - 1;
    }

    float inv_sqrt = inv_sqrt_table[index];
}

// Função para ler dados de um arquivo CSV
float** read_csv(const char* filename, int* num_elements, int* num_dimensions) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    // Determine the number of elements and dimensions
    *num_elements = 0;
    *num_dimensions = 0;
    char line[1024];
    while (fgets(line, sizeof(line), file)) {
        if (*num_elements == 0) {
            char* token = strtok(line, ",");
            while (token) {
                (*num_dimensions)++;
                token = strtok(NULL, ",");
            }
        }
        (*num_elements)++;
    }
    rewind(file);

    // Allocate memory for the features
    float** features = (float**)malloc(*num_elements * sizeof(float*));
    for (int i = 0; i < *num_elements; i++) {
        features[i] = (float*)malloc(*num_dimensions * sizeof(float));
    }

    // Read the data
    int i = 0;
    
    while (fgets(line, sizeof(line), file)) {
        int j = 0;
        char* token = strtok(line, ",");
        while (token) {
            features[i][j++] = atof(token);
            token = strtok(NULL, ",");
        }
        i++;
    }

    fclose(file);
    return features;
}

// Função para medir o tempo de execução usando a biblioteca 'resources'
void get_resource_usage(struct rusage* usage) {
    getrusage(RUSAGE_SELF, usage);
}

void print_resource_usage(const char* label, struct rusage* usage) {
    printf("%s\n", label);
    printf("User time: %ld.%06ld seconds\n", usage->ru_utime.tv_sec, usage->ru_utime.tv_usec);
    printf("System time: %ld.%06ld seconds\n", usage->ru_stime.tv_sec, usage->ru_stime.tv_usec);
    printf("Maximum resident set size: %ld kilobytes\n", usage->ru_maxrss);
}

int main() {
    int num_elements, num_dimensions;
    float** features = read_csv("data.csv", &num_elements, &num_dimensions);

    struct rusage start_usage, end_usage;
    
    get_resource_usage(&start_usage);
    for (int i = 0; i < num_elements; i++) {
        normalize_feature_vector(features[i], num_dimensions);
    }
    get_resource_usage(&end_usage);

    printf("Normalized features:\n");
    for (int i = 0; i < num_elements; i++) {
        for (int j = 0; j < num_dimensions; j++) {
            printf("%f ", features[i][j]);
        }
        printf("\n");
    }

    printf("Execution time and resource usage:\n");
    print_resource_usage("Start Usage", &start_usage);
    print_resource_usage("End Usage", &end_usage);

    // Free allocated memory
    for (int i = 0; i < num_elements; i++) {
        free(features[i]);
    }
    free(features);

    return 0;
}