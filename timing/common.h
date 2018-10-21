#include <vector>
#include <set>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <utility>

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdlib>


using namespace std;

int Ks[] = {1, 10, 100, -1};

float *get_data(const char *file, size_t dim, size_t *n) {
    struct stat sb;
    stat(file, &sb);
    size_t N = sb.st_size / (sizeof(float) * dim);
    *n = N;

    float *data = new float[N * dim];

    FILE *fd;
    fd = fopen(file, "rb");
    fread(data, sizeof(float), N * dim, fd);
    fclose(fd);

    return data;
}

float *read_memory(const char *file, size_t n, size_t dim) {
    float *data = new float[n * dim];

    struct stat sb;
    stat(file, &sb);

    if(sb.st_size != n * dim * sizeof(float)) {
        std::cerr << "Size of the file is " << sb.st_size << ", while the expected size is: " << n * dim * sizeof(float) << "\n";
        return NULL;
    }

    FILE *fd;
    if ((fd = fopen(file, "rb")) == NULL) {
        std::cerr << "Could not open file " << file << " for reading.\n";
        return NULL;
    }

    size_t read = fread(data, sizeof(float), n * dim, fd);
    if (read != n * dim) {
        std::cerr << "Expected size of the read was " << n * dim << ", but " << read << " was read.\n";
        return NULL;
    }

    fclose(fd);
    return data;
}

float *read_mmap(const char *file, size_t n, size_t dim) {
    FILE *fd;
    if ((fd = fopen(file, "rb")) == NULL)
        return NULL;

    float *data;

    if ((data = reinterpret_cast<float *> (
#ifdef MAP_POPULATE
            mmap(0, n * dim * sizeof(float), PROT_READ,
            MAP_SHARED | MAP_POPULATE, fileno(fd), 0))) == MAP_FAILED) {
#else
            mmap(0, n * dim * sizeof(float), PROT_READ,
            MAP_SHARED, fileno(fd), 0))) == MAP_FAILED) {
#endif
            return NULL;
    }

    fclose(fd);
    return data;
}


void results(int k, const vector<double> &times, const vector<set<int>> &idx, const char *truth, bool verbose,
             std::vector<double> projection_times = std::vector<double>(),
             std::vector<double> exact_times = std::vector<double>(),
             std::vector<double> elect_times = std::vector<double>()) {
    double time;
    vector<set<int>> correct;

    ifstream fs(truth);
    for (int j = 0; fs >> time; ++j) {
        set<int> res;
        for (int i = 0; i < k; ++i) {
            int r;
            fs >> r;
            res.insert(r);
        }
        correct.push_back(res);
    }

    vector<pair<double, double>> results;

    double accuracy, total_time = 0, total_accuracy = 0;
    double projection_time = 0, exact_time = 0, elect_time = 0;
    bool extra_times = !projection_times.empty();
    for (unsigned i = 0; i < times.size(); ++i) {
        set<int> intersect;
        set_intersection(correct[i].begin(), correct[i].end(), idx[i].begin(), idx[i].end(),
                         inserter(intersect, intersect.begin()));
        accuracy = intersect.size() / static_cast<double>(k);
        total_time += times[i];

        if(extra_times) {
          projection_time += projection_times[i];
          exact_time += exact_times[i];
          elect_time += elect_times[i];
        }

        total_accuracy += accuracy;
        results.push_back(make_pair(times[i], accuracy));
    }

    double mean_accuracy = total_accuracy / results.size(), variance = 0;
    for (auto res : results)
        variance += (res.second - mean_accuracy) * (res.second - mean_accuracy);
    variance /= (results.size() - 1);

    cout << setprecision(5);
    if(verbose){
      cout << "accuracy: " << mean_accuracy
           << ", variance:  " << variance
           << ", query time: " << total_time;
           if(extra_times) cout << ", projection time: " << projection_time
                                << ", exact search time: " << exact_time
                                << ", elect time: " << elect_time;
    } else {
      cout << mean_accuracy << " "
           << variance << " "
           << total_time << " ";
           if(extra_times) cout << projection_time << " "
                                << exact_time << " "
                                << elect_time << " ";
    }

}
