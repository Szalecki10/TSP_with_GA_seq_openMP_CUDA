#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <ctime>
#include <map>
#include <set>
#include <omp.h>

// Includes for CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h> // For blockIdx, threadIdx etc.

// Parametry algorytmu genetycznego
const int POPULATION_SIZE = 1500;
const int NUM_GENERATIONS = 5000;
const double MUTATION_RATE = 0.02;
const int DOMINANCE_THRESHOLD = 100;
const int ELITE_SIZE = 8;
const int TOURNAMENT_SIZE = 10;

// Struktura reprezentująca miasto
// Jest to prosty POD (Plain Old Data), więc może być używany bezpośrednio z CUDA.
struct City {
    int id;
    double x, y;
};

// Struktura reprezentująca trasę (permutację miast)
using Tour = std::vector<int>;

// Makro do sprawdzania błędów CUDA
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Funkcja do obliczania odległości między dwoma miastami (wersja dla CPU)
double host_distance(const City& c1, const City& c2) {
    return std::sqrt(std::pow(c1.x - c2.x, 2) + std::pow(c1.y - c2.y, 2));
}

// Funkcja do obliczania odległości między dwoma miastami (wersja dla GPU)
__device__ double device_distance(const City& c1, const City& c2) {
    double dx = c1.x - c2.x;
    double dy = c1.y - c2.y;
    return sqrt(dx * dx + dy * dy); // sqrtf dla float, sqrt dla double
}

// Funkcja do obliczania długości trasy (wersja dla CPU)
double calculate_tour_length_cpu(const Tour& tour, const std::vector<City>& cities) {
    double length = 0.0;
    int n = tour.size();
    for (int i = 0; i < n; ++i) {
        length += host_distance(cities[tour[i] - 1], cities[tour[(i + 1) % n] - 1]);
    }
    return length;
}

// Kernel CUDA do obliczania długości tras
__global__ void calculate_tour_length_kernel(const City* d_cities, const int* d_flat_population, double* d_tour_lengths, int num_cities, int current_population_size) {
    int tour_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (tour_idx < current_population_size) {
        double length = 0.0;
        // Wskaźnik na początek bieżącej trasy w spłaszczonej populacji
        const int* current_tour_ptr = d_flat_population + tour_idx * num_cities;

        for (int i = 0; i < num_cities; ++i) {
            // ID miast są 1-bazowe w trasie, tablica d_cities jest 0-indeksowa
            City c1 = d_cities[current_tour_ptr[i] - 1];
            City c2 = d_cities[current_tour_ptr[(i + 1) % num_cities] - 1];
            length += device_distance(c1, c2);
        }
        d_tour_lengths[tour_idx] = length;
    }
}

// Funkcja do wczytywania danych z pliku TSPLIB
std::vector<City> load_tsp_data(const std::string& filename) {
    std::vector<City> cities;
    std::ifstream file(filename);
    std::string line;
    bool reading_coords = false;
    while (std::getline(file, line)) {
        if (line.find("NODE_COORD_SECTION") != std::string::npos) {
            reading_coords = true;
            continue;
        }
        if (line.find("EOF") != std::string::npos) {
            break;
        }
        if (reading_coords) {
            std::stringstream ss(line);
            int id;
            double x, y;
            if (ss >> id >> x >> y) {
                cities.push_back({id, x, y});
            }
        }
    }
    return cities;
}

// Funkcja do generowania losowej trasy
Tour generate_random_tour(int num_cities) {
    Tour tour(num_cities);
    for (int i = 0; i < num_cities; ++i) {
        tour[i] = i + 1;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(tour.begin(), tour.end(), gen);
    return tour;
}

// Funkcja do generowania populacji początkowej metodą zachłanną
std::vector<Tour> generate_greedy_population(int population_size_param, int num_cities, const std::vector<City>& cities) {
    std::vector<Tour> population(population_size_param);
    #pragma omp parallel for
    for (int i = 0; i < population_size_param; ++i) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> start_node_dist(1, num_cities);

        int start_node = start_node_dist(gen);
        Tour current_tour;
        std::vector<bool> visited(num_cities + 1, false);
        current_tour.push_back(start_node);
        visited[start_node] = true;

        int current_city_id = start_node;
        for (int j = 1; j < num_cities; ++j) {
            int next_city_id = -1;
            double min_dist_val = std::numeric_limits<double>::max();
            for (int k = 1; k <= num_cities; ++k) {
                if (!visited[k]) {
                    double dist = host_distance(cities[current_city_id - 1], cities[k - 1]);
                    if (dist < min_dist_val) {
                        min_dist_val = dist;
                        next_city_id = k;
                    }
                }
            }
            if (next_city_id != -1) {
                current_tour.push_back(next_city_id);
                visited[next_city_id] = true;
                current_city_id = next_city_id;
            } else { // Should not happen in a connected graph if not all cities visited
                for (int k = 1; k <= num_cities; ++k) {
                    if (!visited[k]) {
                        current_tour.push_back(k);
                        visited[k] = true; // Mark as visited
                    }
                }
                break; // Exit loop once all remaining are added
            }
        }
        population[i] = current_tour;
    }
    return population;
}

// Funkcja pomocnicza do spłaszczania populacji dla GPU
std::vector<int> flatten_population(const std::vector<Tour>& population, int num_cities) {
    if (population.empty()) return {};
    std::vector<int> flat_population(population.size() * num_cities);
    for (size_t i = 0; i < population.size(); ++i) {
        for (int j = 0; j < num_cities; ++j) {
            if (j < population[i].size()) { // Safety check
                 flat_population[i * num_cities + j] = population[i][j];
            } else {
                // Handle error or fill with a placeholder if tours can have variable lengths (not typical for TSP GA)
                // For TSP, all tours should have num_cities length.
            }
        }
    }
    return flat_population;
}


// Funkcja do oceny populacji (zwraca pary: trasa, długość) - wersja CUDA
std::vector<std::pair<Tour, double>> evaluate_population_cuda(
    const std::vector<Tour>& current_population,
    const std::vector<City>& host_cities, // Używane do rekonstrukcji par, ale nie do obliczeń
    City* d_cities,                       // Wskaźnik na miasta w pamięci GPU
    int num_cities) {

    int current_pop_size = current_population.size();
    if (current_pop_size == 0) return {};

    // 1. Spłaszcz populację hosta dla GPU
    std::vector<int> flat_host_population = flatten_population(current_population, num_cities);

    // 2. Alokuj pamięć GPU dla spłaszczonej populacji i długości tras
    int* d_flat_population;
    double* d_tour_lengths;
    CUDA_CHECK(cudaMalloc(&d_flat_population, flat_host_population.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tour_lengths, current_pop_size * sizeof(double)));

    // 3. Skopiuj spłaszczoną populację z hosta na GPU
    CUDA_CHECK(cudaMemcpy(d_flat_population, flat_host_population.data(), flat_host_population.size() * sizeof(int), cudaMemcpyHostToDevice));

    // 4. Skonfiguruj i uruchom kernel CUDA
    // d_cities powinno być już skopiowane na GPU na początku main()
    int threads_per_block = 256;
    int blocks_per_grid = (current_pop_size + threads_per_block - 1) / threads_per_block;

    calculate_tour_length_kernel<<<blocks_per_grid, threads_per_block>>>(d_cities, d_flat_population, d_tour_lengths, num_cities, current_pop_size);
    CUDA_CHECK(cudaGetLastError());      // Sprawdź błędy uruchomienia kernela
    CUDA_CHECK(cudaDeviceSynchronize()); // Poczekaj na zakończenie kernela

    // 5. Skopiuj obliczone długości tras z GPU na hosta
    std::vector<double> host_tour_lengths(current_pop_size);
    CUDA_CHECK(cudaMemcpy(host_tour_lengths.data(), d_tour_lengths, current_pop_size * sizeof(double), cudaMemcpyDeviceToHost));

    // 6. Zwolnij pamięć GPU (d_cities jest zarządzane zewnętrznie)
    CUDA_CHECK(cudaFree(d_flat_population));
    CUDA_CHECK(cudaFree(d_tour_lengths));

    // 7. Zrekonstruuj ocenioną populację (łączenie oryginalnych tras z nowymi długościami) i posortuj
    std::vector<std::pair<Tour, double>> evaluated_population_pairs(current_pop_size);
    for (int i = 0; i < current_pop_size; ++i) {
        evaluated_population_pairs[i] = {current_population[i], host_tour_lengths[i]};
    }

    std::sort(evaluated_population_pairs.begin(), evaluated_population_pairs.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    return evaluated_population_pairs;
}


// Funkcja do selekcji turniejowej
Tour tournament_selection(const std::vector<std::pair<Tour, double>>& evaluated_population) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, evaluated_population.size() - 1);
    std::vector<std::pair<Tour, double>> contenders;
    for (int i = 0; i < TOURNAMENT_SIZE; ++i) {
        contenders.push_back(evaluated_population[dist(gen)]);
    }
    std::sort(contenders.begin(), contenders.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });
    return contenders[0].first;
}

// Funkcja do crossover PMX
Tour pmx_crossover(const Tour& parent1, const Tour& parent2) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, parent1.size() - 1);
    int p1 = dist(gen);
    int p2 = dist(gen);
    if (p1 > p2) std::swap(p1, p2);

    Tour child1 = parent1;
    Tour child2 = parent2;

    std::map<int, int> mapping1, mapping2;

    for (int i = p1; i <= p2; ++i) {
        child1[i] = parent2[i];
        child2[i] = parent1[i];
        mapping1[parent2[i]] = parent1[i];
        mapping2[parent1[i]] = parent2[i];
    }

    for (size_t i = 0; i < parent1.size(); ++i) {
        if (i < p1 || i > p2) {
            int gene1 = child1[i];
            while (mapping1.count(gene1)) {
                gene1 = mapping1[gene1];
            }
            child1[i] = gene1;

            int gene2 = child2[i];
            while (mapping2.count(gene2)) {
                gene2 = mapping2[gene2];
            }
            child2[i] = gene2;
        }
    }
    // std::rand() jest mniej preferowany niż <random> w nowoczesnym C++
    // ale dla zachowania oryginalnej logiki, jeśli ma to znaczenie:
    // std::uniform_int_distribution<> coin_flip(0, 1);
    // return (coin_flip(gen) == 0) ? child1 : child2;
    return (std::rand() % 2 == 0) ? child1 : child2; // Zachowujemy oryginalny wybór, jeśli był zamierzony
}

// Funkcja do mutacji inwersyjnej
void inversion_mutation(Tour& tour) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> prob(0.0, 1.0);
    std::uniform_int_distribution<> index_dist(0, tour.size() - 1);

    if (prob(gen) < MUTATION_RATE) {
        if (tour.size() < 2) return; // Inwersja wymaga co najmniej 2 elementów
        int i = index_dist(gen);
        int j = index_dist(gen);
        if (i == j) { // Jeśli punkty są te same, spróbuj wybrać inny j, aby uniknąć pustego zakresu
           j = (j + 1) % tour.size(); // Prosta strategia, można ulepszyć
        }
        if (i > j) std::swap(i, j);
        std::reverse(tour.begin() + i, tour.begin() + j + 1);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Uzycie: " << argv[0] << " <nazwa_pliku_tsp>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    std::vector<City> cities_vec = load_tsp_data(filename);
    int num_cities = cities_vec.size();

    if (num_cities == 0) {
        std::cerr << "Nie udalo sie wczytac danych miast." << std::endl;
        return 1;
    }

    // Inicjalizacja CUDA i alokacja pamięci na GPU dla miast
    City* d_cities;
    CUDA_CHECK(cudaMalloc(&d_cities, num_cities * sizeof(City)));
    CUDA_CHECK(cudaMemcpy(d_cities, cities_vec.data(), num_cities * sizeof(City), cudaMemcpyHostToDevice));


    double start_time = omp_get_wtime();
    std::srand(static_cast<unsigned>(time(0))); // Dla std::rand() w pmx_crossover

    std::vector<Tour> population = generate_greedy_population(POPULATION_SIZE, num_cities, cities_vec);
    std::vector<std::pair<Tour, double>> evaluated_population = evaluate_population_cuda(population, cities_vec, d_cities, num_cities);
    std::pair<Tour, double> global_best_solution = evaluated_population[0];
    int stagnation_counter = 0;

    for (int generation = 0; generation < NUM_GENERATIONS; ++generation) {
        // double generation_start_time = omp_get_wtime(); // Można odkomentować do profilowania generacji

        // Sortowanie już wykonane w evaluate_population_cuda
        // std::sort(evaluated_population.begin(), evaluated_population.end(), [](const auto& a, const auto& b) {
        //     return a.second < b.second;
        // });
        std::pair<Tour, double> current_best_solution_in_gen = evaluated_population[0];

        if (current_best_solution_in_gen.second < global_best_solution.second) {
            global_best_solution = current_best_solution_in_gen;
            stagnation_counter = 0;
             std::cout << "Pokolenie " << generation << ", Najlepsza dlugosc trasy: " << global_best_solution.second
                       // << ", Czas generacji: " << omp_get_wtime() - generation_start_time << " s" // Jeśli generation_start_time jest aktywne
                       << std::endl;
        } else {
            stagnation_counter++;
            if (stagnation_counter > DOMINANCE_THRESHOLD) {
                std::cout << "Dominacja wykryta. Czésciowy restart populacji." << std::endl;
                std::vector<Tour> new_population_on_stagnation;
                for (int i = 0; i < ELITE_SIZE && i < evaluated_population.size(); ++i) {
                    new_population_on_stagnation.push_back(evaluated_population[i].first);
                }
                // Dopełnienie populacji po stagnacji
                int current_elite_fill = new_population_on_stagnation.size();
                std::vector<Tour> new_greedy_elements = generate_greedy_population(POPULATION_SIZE - current_elite_fill, num_cities, cities_vec);
                new_population_on_stagnation.insert(new_population_on_stagnation.end(), new_greedy_elements.begin(), new_greedy_elements.end());
                
                // Upewnij się, że rozmiar jest prawidłowy, jeśli generate_greedy_population zwróci mniej/więcej
                if(new_population_on_stagnation.size() > POPULATION_SIZE) {
                    new_population_on_stagnation.resize(POPULATION_SIZE);
                }
                while(new_population_on_stagnation.size() < POPULATION_SIZE){
                     new_population_on_stagnation.push_back(generate_random_tour(num_cities));
                }

                population = new_population_on_stagnation;
                evaluated_population = evaluate_population_cuda(population, cities_vec, d_cities, num_cities);
                stagnation_counter = 0;
                if(evaluated_population.empty()){ // Safety break
                    std::cerr << "Populacja pusta po restarcie stagnacji!" << std::endl;
                    break;
                }
                global_best_solution = evaluated_population[0]; // Aktualizuj global_best_solution po restarcie
                continue;
            }
        }

        std::vector<Tour> next_generation_population;
        // Elityzm: zachowaj ELITE_SIZE najlepszych osobników
        for (int i = 0; i < ELITE_SIZE && i < evaluated_population.size(); ++i) {
            next_generation_population.push_back(evaluated_population[i].first);
        }

        std::set<std::vector<int>> unique_offspring_set; // Używane do śledzenia unikalności na CPU
        // Wypełnij resztę populacji potomstwem
        // Ta pętla OpenMP jest dla generowania potomstwa na CPU
        std::vector<Tour> offspring_buffer(POPULATION_SIZE - next_generation_population.size());

        #pragma omp parallel
        {
            std::random_device rd_local;
            std::mt19937 gen_local(rd_local()); // Lokalny generator dla wątku

            #pragma omp for
            for (int i = 0; i < offspring_buffer.size(); ++i) {
                Tour parent1 = tournament_selection(evaluated_population);
                Tour parent2 = tournament_selection(evaluated_population);
                Tour child = pmx_crossover(parent1, parent2);
                inversion_mutation(child); // Mutacja używa własnego std::random_device
                offspring_buffer[i] = child;
            }
        }
        
        // Dodaj potomstwo do następnej generacji, sprawdzając unikalność (opcjonalne, ale było w oryginale)
        for(const auto& child_tour : offspring_buffer){
            if(next_generation_population.size() >= POPULATION_SIZE) break;
            // Można dodać sprawdzanie unikalności, jeśli jest to krytyczne
            // if(unique_offspring_set.find(child_tour) == unique_offspring_set.end()){
            //    next_generation_population.push_back(child_tour);
            //    unique_offspring_set.insert(child_tour);
            // }
            next_generation_population.push_back(child_tour); // Prostsze dodawanie
        }


        // Jeśli wygenerowano za mało potomków (np. przez unikalność lub mały ELITE_SIZE), dopełnij losowymi/zachłannymi trasami
        while (next_generation_population.size() < POPULATION_SIZE) {
             if (std::rand() % 2 == 0) { // Użycie std::rand jak w oryginalnym kodzie
                next_generation_population.push_back(generate_random_tour(num_cities));
            } else {
                // Generuj jedną trasę zachłanną - uproszczona wersja z generate_greedy_population
                std::vector<Tour> one_greedy_tour_vec = generate_greedy_population(1, num_cities, cities_vec);
                if(!one_greedy_tour_vec.empty()){
                    next_generation_population.push_back(one_greedy_tour_vec[0]);
                } else { // fallback
                     next_generation_population.push_back(generate_random_tour(num_cities));
                }
            }
        }
        if(next_generation_population.size() > POPULATION_SIZE){
            next_generation_population.resize(POPULATION_SIZE);
        }


        population = next_generation_population;
        if(population.empty()){
             std::cerr << "Populacja stala sie pusta przed ewaluacja w generacji " << generation << std::endl;
             break;
        }
        evaluated_population = evaluate_population_cuda(population, cities_vec, d_cities, num_cities);
        if(evaluated_population.empty()){
             std::cerr << "Ewaluacja zwrocila pusta populacje w generacji " << generation << std::endl;
             break;
        }
    }

    double end_time = omp_get_wtime();
    double total_time = end_time - start_time;

    std::cout << "\nNajlepsza znaleziona trasa: ";
    for (int city_id : global_best_solution.first) {
        std::cout << city_id << " ";
    }
    std::cout << "\nDlugosc trasy: " << global_best_solution.second << std::endl;
    std::cout << "Czas calkowitego wykonania (wall clock): " << total_time << " s" << std::endl;

    // Zwolnij pamięć GPU dla miast
    CUDA_CHECK(cudaFree(d_cities));

    return 0;
}