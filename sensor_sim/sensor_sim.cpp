// Simulated field sensors. Prints one JSON array (one reading per zone).
// Build:  g++ -O2 -o sensor_sim sensor_sim.cpp
// Run:    ./sensor_sim 6
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

int main(int argc, char** argv) {
    int zones = argc > 1 ? std::atoi(argv[1]) : 6;
    std::mt19937 g(std::random_device{}());
    std::normal_distribution<double> pressure(4.0, 0.12), flow(100, 5),
        temp(35, 3), hum(25, 6), moist(22, 8);
    std::uniform_real_distribution<double> u(0, 1);

    std::cout << std::fixed << std::setprecision(2) << "[";
    for (int i = 0; i < zones; i++) {
        double p = pressure(g), f = flow(g);
        if (u(g) < 0.12) {          // simulated leak: pressure drop + flow surge
            p -= 1.2 + u(g);
            f += 25 + 30 * u(g);
        }
        double m = std::clamp(moist(g), 4.0, 45.0);
        double h = std::clamp(hum(g), 8.0, 70.0);
        std::cout << (i ? "," : "") << "{\"zone\":" << i
                  << ",\"pressure\":" << p << ",\"flow\":" << f
                  << ",\"temp\":" << temp(g) << ",\"humidity\":" << h
                  << ",\"moisture\":" << m << "}";
    }
    std::cout << "]\n";
}
