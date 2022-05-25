#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <random>

#define H 100
#define W 100
#define PIXEL_N H * W
#define CHANNEL 3
#define GAMMA_S 1.0 / (1 * H * W)
#define GAMMA_C 1.0 / (1 * 255 * 255)
#define x first
#define y second

using namespace std;

typedef void (*fptr_t)(vector<vector<int> > &);

int K;
string filename;

void random_cluster(vector<vector<int> > &cluster);
void kmeans__(vector<vector<int> > &cluster);
int distance(vector<int> &v1, vector<int> &v2);

// fptr_t init;
map<string, fptr_t> init_func = { {"random", random_cluster}, {"kmeans++", kmeans__} };

void random_cluster(vector<vector<int> > &cluster) {
    srand(time(0));
    for (int i = 0; i < PIXEL_N; i++) {
        cluster[rand() % K].push_back(i);
    }
}

vector<int> get_distances_2(pair<int, int> centroid) {
    vector<int> distances;
    vector<int> v1{centroid.x, centroid.y};
    for (int i = 0; i < PIXEL_N; i++) {
        int x = i / W, y = i % H;
        vector<int> v2{x, y};
        int dist = distance(v1, v2);
        distances.push_back(pow(dist, 2));
    }
    return distances;
}

pair<int, int> next_centroid(pair<int, int> centroid) {
    random_device rd;
    mt19937 gen(rd());
    vector<int> distances = get_distances_2(centroid);
    discrete_distribution<> d(distances.begin(), distances.end());
    int p = d(gen);
    return make_pair(p / W, p % H);
}

void kmeans__(vector<vector<int> > &cluster) {
    srand(time(0));
    vector<pair<int, int> > centroids(K);
    centroids[0].x = rand() % W, centroids[0].y = rand() % H;

    for (int c = 1; c < K; c++) {
        centroids[c] = next_centroid(centroids[c - 1]);
    }

    for (int i = 0; i < PIXEL_N; i++) {
        vector<int> dist(K);
        vector<int> v1{i / W, i % H};
        for (int c = 0; c < K; c++) {
            vector<int> v2{centroids[c].x, centroids[c].y};
            dist[c] = distance(v1, v2);
        }
        int min_dist_c = min_element(dist.begin(), dist.end()) - dist.begin();
        cluster[min_dist_c].push_back(i);
    }
}

vector<vector<int> > text2image() {
    vector<vector<int> > image;
    image.resize(PIXEL_N);

    ifstream ifs(filename);
    for (int i = 0; i < image.size(); i++) {
        image[i].resize(CHANNEL);
        ifs >> image[i][0] >> image[i][1] >> image[i][2];
    }

    return image;
}

int distance(vector<int> &v1, vector<int> &v2) {
    int dist = 0;
    for (int i = 0; i < v1.size(); i++) {
        dist += pow(v1[i] - v2[i], 2);
    }
    return dist;
}

vector<vector<float> > kernel(vector<vector<int> > &image) {
    vector<vector<float> > k(PIXEL_N);
    for (int i = 0; i < k.size(); i++) {
        k[i].resize(PIXEL_N);
    }
    
    for (int i = 0; i < image.size(); i++) {
        int x = i / W, y = i % H;
        vector<int> coor{x, y};
        for (int j = i; j < image.size(); j++) {
            int _x = j / W, _y = j % H;
            vector<int> _coor{_x, _y};
            int dS = distance(coor, _coor); // spatial 
            int dC = distance(image[i], image[j]); // color
            k[i][j] = exp(-GAMMA_S * dS + -GAMMA_C * dC);
            k[j][i] = exp(-GAMMA_S * dS + -GAMMA_C * dC); // symmetric
        }
    }
    return k;
}

float second(vector<vector<float> > &k, vector<vector<int> > &cluster, int i, int c) {
    float second_term = 0;
    for (int j = 0; j < cluster[c].size(); j++) {
        int x = cluster[c][j];
        second_term += k[i][x];
    }
    return 2 * second_term / cluster[c].size();
}

vector<float> third(vector<vector<float> > &k, vector<vector<int> > &cluster) {
    vector<float> third_vec(K, 0);
    for (int c = 0; c < K; c++) {
        for (int j = 0; j < cluster[c].size(); j++) {
            for (int l = j + 1; l < cluster[c].size(); l++) {
                int x = cluster[c][j], y = cluster[c][l];
                third_vec[c] += k[x][y];
            }
        }
        if (cluster[c].size() != 0) {
            third_vec[c] /= pow(cluster[c].size(), 2);
        }
    }
    cout << "third: \n";
    for (int i = 0; i < K; i++) {
        cout << cluster[i].size() << '\n';
    }
    return third_vec;
}

void cluster2table(vector<vector<int> > &cluster, vector<int> &ctable) {
    for (int c = 0; c < K; c++) {
        for (int i = 0; i < cluster[c].size(); i++) {
            ctable[cluster[c][i]] = c;
        }
    }
}

void clustering(vector<vector<float> > &k, vector<vector<int> > &cluster, vector<vector<int> > &new_cluster) {
    for (int c = 0; c < K; c++) {
        new_cluster[c].clear();
    }
    
    vector<float> third_vec = third(k, cluster);
    for (int i = 0; i < PIXEL_N; i++) {
        vector<float> dist(K);
        for (int c = 0; c < K; c++) {
            dist[c] = k[i][i] - second(k, cluster, i, c) + third_vec[c];
        }
        int min_dist_c = min_element(dist.begin(), dist.end()) - dist.begin();
        new_cluster[min_dist_c].push_back(i);
    }
}

bool isdiff(vector<vector<int> > &cluster, vector<vector<int> > &new_cluster) {
    for (int c = 0; c < K; c++) {
        if (cluster[c].size() != new_cluster[c].size()) {
            // cout << "size diff: c = " << c << ", size = " << cluster[c].size() << ' ' << new_cluster[c].size() << '\n';
            return true;
        }
        for (int i = 0; i < cluster[c].size(); i++) {
            if (cluster[c][i] != new_cluster[c][i]) {
                // cout << "value diff: c = " << c << ", i = " << i << ", value = " << cluster[c][i] << ' ' << new_cluster[c][i] << '\n'; 
                return true;
            }
        }
    }
    return false;
}

void output(vector<int> &ctable, int iter) {
    string text_filename = filename.substr(0, filename.size() - 4) + "_table_" + to_string(iter) + ".txt";
    cout << text_filename << '\n';
    ofstream ofs(text_filename);
    for (int i = 0; i < PIXEL_N; i++) {
        ofs << ctable[i] << '\n';
    }
    ofs.close();
}

void kernel_k_means(vector<vector<float> > k, string init_method) {
    vector<vector<int> > cluster(K), new_cluster(K);
    vector<int> ctable(PIXEL_N);

    int iter = 0;

    init_func[init_method](cluster);
    cluster2table(cluster, ctable);
    output(ctable, iter);

    while (isdiff(cluster, new_cluster)) {
        iter++;
        clustering(k, cluster, new_cluster);
        cluster.swap(new_cluster);
        cluster2table(cluster, ctable);
        output(ctable, iter);
    }
}

int main(int argc, char *argv[]) {
    K = atoi(argv[1]);
    filename = string(argv[3]);
    vector<vector<int> > image = text2image();
    vector<vector<float> > k = kernel(image);
    kernel_k_means(k, string(argv[2]));
}