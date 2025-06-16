#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <ctime> // Pozostawione dla std::time(nullptr) jako seed, ale OpenMP użyje omp_get_wtime()
#include <map>
#include <set>
#include <omp.h> // Dodano OpenMP

// Parametry algorytmu genetycznego
const int POPULATION_SIZE = 1500;
const int NUM_GENERATIONS = 5000;
const double MUTATION_RATE = 0.02;
const int DOMINANCE_THRESHOLD = 100;
const int ELITE_SIZE = 8;
const int TOURNAMENT_SIZE = 10;

// Struktura reprezentująca miasto
struct City {
    int id;
    double x, y;
};

// Struktura reprezentująca trasę (permutację miast)
using Tour = std::vector<int>;

// Funkcja do obliczania odległości między dwoma miastami
double distance(const City& c1, const City& c2) {
    return std::sqrt(std::pow(c1.x - c2.x, 2) + std::pow(c1.y - c2.y, 2));
}

// Funkcja do obliczania długości trasy
double calculate_tour_length(const Tour& tour, const std::vector<City>& cities) {
    double length = 0.0;
    int n = tour.size();
    for (int i = 0; i < n; ++i) {
        length += distance(cities[tour[i] - 1], cities[tour[(i + 1) % n] - 1]);
    }
    return length;
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

// Funkcja do generowania losowej trasy (z generatorem)
Tour generate_random_tour(int num_cities, std::mt19937& gen) {
    Tour tour(num_cities);
    for (int i = 0; i < num_cities; ++i) {
        tour[i] = i + 1;
    }
    std::shuffle(tour.begin(), tour.end(), gen);
    return tour;
}

// Funkcja do generowania populacji początkowej metodą zachłanną (prosta wersja)
std::vector<Tour> generate_greedy_population(int population_size, int num_cities, const std::vector<City>& cities) {
    std::vector<Tour> population(population_size);
    unsigned int base_seed = std::random_device{}(); // Pojedyncze ziarno bazowe

    #pragma omp parallel
    {
        std::mt19937 gen(base_seed + omp_get_thread_num()); // Każdy wątek ma własny generator z unikalnym ziarnem
        std::uniform_int_distribution<> start_node_dist(1, num_cities);

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < population_size; ++i) {
            int start_node = start_node_dist(gen);
            Tour current_tour;
            current_tour.reserve(num_cities);
            std::vector<bool> visited(num_cities + 1, false);
            current_tour.push_back(start_node);
            visited[start_node] = true;

            int current_city_val = start_node;
            for (int j = 1; j < num_cities; ++j) {
                int next_city = -1;
                double min_dist_val = std::numeric_limits<double>::max();
                for (int k = 1; k <= num_cities; ++k) {
                    if (!visited[k]) {
                        double dist_val = distance(cities[current_city_val - 1], cities[k - 1]);
                        if (dist_val < min_dist_val) {
                            min_dist_val = dist_val;
                            next_city = k;
                        }
                    }
                }
                if (next_city != -1) {
                    current_tour.push_back(next_city);
                    visited[next_city] = true;
                    current_city_val = next_city;
                } else {
                    for (int k = 1; k <= num_cities; ++k) {
                        if (!visited[k]) {
                            current_tour.push_back(k);
                            visited[k] = true; // Upewnij się, że jest oznaczone
                        }
                    }
                    break;
                }
            }
            population[i] = current_tour;
        }
    }
    return population;
}

// Funkcja do oceny populacji (zwraca pary: trasa, długość)
std::vector<std::pair<Tour, double>> evaluate_population(const std::vector<Tour>& population, const std::vector<City>& cities) {
    std::vector<std::pair<Tour, double>> evaluated_population(population.size());

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < population.size(); ++i) {
        evaluated_population[i] = {population[i], calculate_tour_length(population[i], cities)};
    }

    std::sort(evaluated_population.begin(), evaluated_population.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });
    return evaluated_population;
}

// Funkcja do selekcji turniejowej (z generatorem)
Tour tournament_selection(const std::vector<std::pair<Tour, double>>& evaluated_population, std::mt19937& gen) {
    std::uniform_int_distribution<> dist(0, evaluated_population.size() - 1);
    std::vector<std::pair<Tour, double>> contenders;
    contenders.reserve(TOURNAMENT_SIZE);
    for (int i = 0; i < TOURNAMENT_SIZE; ++i) {
        contenders.push_back(evaluated_population[dist(gen)]);
    }
    std::sort(contenders.begin(), contenders.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });
    return contenders[0].first;
}

// Funkcja do crossover PMX (z generatorem)
Tour pmx_crossover(const Tour& parent1, const Tour& parent2, std::mt19937& gen) {
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
        if (i < (size_t)p1 || i > (size_t)p2) {
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
    std::uniform_int_distribution<> coin_flip(0, 1);
    return (coin_flip(gen) == 0) ? child1 : child2;
}

// Funkcja do mutacji inwersyjnej (z generatorem)
void inversion_mutation(Tour& tour, std::mt19937& gen) {
    std::uniform_real_distribution<> prob(0.0, 1.0);
    if (prob(gen) < MUTATION_RATE) {
        if (tour.size() < 2) return; // Potrzebne co najmniej 2 elementy do odwrócenia
        std::uniform_int_distribution<> index_dist(0, tour.size() - 1);
        int i = index_dist(gen);
        int j = index_dist(gen);
        if (i == j) { // Jeśli wylosowano te same indeksy, spróbuj wylosować j jeszcze raz, chyba że rozmiar to 2
            if (tour.size() > 1) { // Dla tour.size() == 1, i tak nie ma co robić
                 j = (i + 1 + index_dist(gen)) % tour.size(); // Spróbuj uzyskać inny j
            }
            if (i == j && tour.size() > 1) { // Jeśli nadal to samo, a można wybrać różne
                 j = (i + 1) % tour.size();
            } else if (tour.size() <=1) return; // Nic nie rób
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
    std::vector<City> cities = load_tsp_data(filename);
    int num_cities = cities.size();

    if (num_cities == 0) {
        std::cerr << "Nie udalo sie wczytac danych miast." << std::endl;
        return 1;
    }

    double total_start_time = omp_get_wtime();
    unsigned int main_seed = std::random_device{}(); // Ziarno dla głównego wątku / operacji sekwencyjnych

    std::vector<Tour> population = generate_greedy_population(POPULATION_SIZE, num_cities, cities);
    std::vector<std::pair<Tour, double>> evaluated_population = evaluate_population(population, cities);
    std::pair<Tour, double> global_best_solution = evaluated_population[0];
    int stagnation_counter = 0;

    for (int generation = 0; generation < NUM_GENERATIONS; ++generation) {
        double generation_start_time = omp_get_wtime();

        // Sortowanie jest sekwencyjne, ale ocena była równoległa
        std::sort(evaluated_population.begin(), evaluated_population.end(), [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
        std::pair<Tour, double> current_best_solution = evaluated_population[0];

        if (current_best_solution.second < global_best_solution.second) {
            global_best_solution = current_best_solution;
            stagnation_counter = 0;
            std::cout << "Pokolenie " << generation << ", Najlepsza dlugosc trasy: " << global_best_solution.second
                      << ", Czas generacji: " << omp_get_wtime() - generation_start_time << " s" << std::endl;
        } else {
            stagnation_counter++;
            if (stagnation_counter > DOMINANCE_THRESHOLD) {
                std::cout << "Dominacja wykryta. Częściowy restart populacji z zachowaniem najlepszych osobnikow." << std::endl;
                std::vector<Tour> new_population_on_stagnation;
                new_population_on_stagnation.reserve(POPULATION_SIZE);
                for (int i = 0; i < ELITE_SIZE; ++i) {
                    new_population_on_stagnation.push_back(evaluated_population[i].first);
                }

                int num_to_regenerate = POPULATION_SIZE - ELITE_SIZE;
                std::vector<Tour> regenerated_individuals(num_to_regenerate);

                #pragma omp parallel
                {
                    std::mt19937 thread_gen(main_seed + omp_get_thread_num() + generation); // Unikalne ziarno
                    std::uniform_int_distribution<> choice_dist(0, 1);
                    std::uniform_int_distribution<> start_node_dist(1, num_cities);

                    #pragma omp for schedule(dynamic)
                    for (int k = 0; k < num_to_regenerate; ++k) {
                        if (choice_dist(thread_gen) == 0) {
                            regenerated_individuals[k] = generate_random_tour(num_cities, thread_gen);
                        } else {
                            // Logika generowania zachłannego dla pojedynczej trasy
                            int start_node = start_node_dist(thread_gen);
                            Tour greedy_tour;
                            greedy_tour.reserve(num_cities);
                            std::vector<bool> visited(num_cities + 1, false);
                            greedy_tour.push_back(start_node);
                            visited[start_node] = true;
                            int current_city_val = start_node;
                            for (int j = 1; j < num_cities; ++j) {
                                int next_city = -1;
                                double min_dist_val = std::numeric_limits<double>::max();
                                for (int city_idx = 1; city_idx <= num_cities; ++city_idx) {
                                    if (!visited[city_idx]) {
                                        double dist_val = distance(cities[current_city_val - 1], cities[city_idx - 1]);
                                        if (dist_val < min_dist_val) {
                                            min_dist_val = dist_val;
                                            next_city = city_idx;
                                        }
                                    }
                                }
                                if (next_city != -1) {
                                    greedy_tour.push_back(next_city);
                                    visited[next_city] = true;
                                    current_city_val = next_city;
                                } else {
                                    for (int city_idx = 1; city_idx <= num_cities; ++city_idx) {
                                        if (!visited[city_idx]) greedy_tour.push_back(city_idx);
                                    }
                                    break;
                                }
                            }
                            regenerated_individuals[k] = greedy_tour;
                        }
                    }
                }
                new_population_on_stagnation.insert(new_population_on_stagnation.end(), regenerated_individuals.begin(), regenerated_individuals.end());
                
                // Upewnij się, że populacja ma właściwy rozmiar (w razie gdyby nie wygenerowano dokładnie tyle ile trzeba)
                std::mt19937 fill_gen_stagnation(main_seed + generation + 1000);
                while(new_population_on_stagnation.size() < POPULATION_SIZE){
                    new_population_on_stagnation.push_back(generate_random_tour(num_cities, fill_gen_stagnation));
                }
                if(new_population_on_stagnation.size() > POPULATION_SIZE) {
                    new_population_on_stagnation.resize(POPULATION_SIZE);
                }


                population = new_population_on_stagnation;
                evaluated_population = evaluate_population(population, cities);
                stagnation_counter = 0;
                continue;
            }
        }

        std::vector<Tour> new_generation_population;
        new_generation_population.reserve(POPULATION_SIZE);

        for (int i = 0; i < ELITE_SIZE; ++i) {
            new_generation_population.push_back(evaluated_population[i].first);
        }

        int num_offspring_to_generate = POPULATION_SIZE - ELITE_SIZE;
        std::vector<Tour> offspring_buffer(num_offspring_to_generate);

        #pragma omp parallel
        {
            std::mt19937 thread_local_gen(main_seed + omp_get_thread_num() + generation + 1); // Unikalne ziarno

            #pragma omp for schedule(dynamic)
            for (int i = 0; i < num_offspring_to_generate; ++i) {
                Tour parent1 = tournament_selection(evaluated_population, thread_local_gen);
                Tour parent2 = tournament_selection(evaluated_population, thread_local_gen);
                Tour child = pmx_crossover(parent1, parent2, thread_local_gen);
                inversion_mutation(child, thread_local_gen);
                offspring_buffer[i] = child;
            }
        }
        
        // Sekwencyjne dodawanie potomstwa z sprawdzaniem unikalności
        std::set<std::vector<int>> unique_offspring_check;
        for(const auto& elite_tour : new_generation_population) {
            unique_offspring_check.insert(elite_tour);
        }

        for (const auto& child : offspring_buffer) {
            if (new_generation_population.size() >= POPULATION_SIZE) break;
            if (unique_offspring_check.find(child) == unique_offspring_check.end()) {
                new_generation_population.push_back(child);
                unique_offspring_check.insert(child);
            }
        }
        
        // Dopełnianie losowymi, jeśli potrzeba
        std::mt19937 fill_gen(main_seed + generation + 2000); // Generator dla dopełniania
        int initial_fill_attempts = 0; // Licznik prób dla unikalnych losowych
        while (new_generation_population.size() < POPULATION_SIZE) {
            Tour random_t = generate_random_tour(num_cities, fill_gen);
            if (unique_offspring_check.find(random_t) == unique_offspring_check.end()) {
                new_generation_population.push_back(random_t);
                unique_offspring_check.insert(random_t);
                initial_fill_attempts = 0; // Resetuj licznik, jeśli znaleziono unikalny
            } else {
                initial_fill_attempts++;
                if (initial_fill_attempts > POPULATION_SIZE * 2 && new_generation_population.size() < POPULATION_SIZE) { 
                    // Jeśli trudno znaleźć unikalny, dodaj nawet jeśli jest duplikatem, aby uniknąć pętli
                    new_generation_population.push_back(random_t); 
                }
            }
             if (new_generation_population.size() >= POPULATION_SIZE && initial_fill_attempts > POPULATION_SIZE * 3) { // Dodatkowy warunek wyjścia
                break;
            }
        }
         if (new_generation_population.size() > POPULATION_SIZE) {
            new_generation_population.resize(POPULATION_SIZE);
        }


        population = new_generation_population;
        evaluated_population = evaluate_population(population, cities);
    }

    double total_end_time = omp_get_wtime();
    double total_time = total_end_time - total_start_time;

    std::cout << "\nNajlepsza znaleziona trasa: ";
    for (int city_id : global_best_solution.first) {
        std::cout << city_id << " ";
    }
    std::cout << "\nDlugosc trasy: " << global_best_solution.second << std::endl;
    std::cout << "Czas calkowitego wykonania (wall clock): " << total_time << " s" << std::endl;

    return 0;
}