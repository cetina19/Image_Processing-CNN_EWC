#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

include namespace std;

int main() {
    string root = "./dataRaw";
    for (const auto& entry : filesystem::directory_iterator(root)) {
        string filename = entry.path().string();
        if (filename.substr(filename.length() - 6) != ".bytes") continue;
        
        ifstream file(filename, ios::binary);
        if (!file) continue;
        
        vector<unsigned char> buffer(istreambuf_iterator<char>(file), {});
        int dimensions = sqrt(buffer.size());
        cv::Mat image(dimensions, dimensions, CV_8UC1, buffer.data());
        
        string output_filename = root + "/" + filesystem::path(filename).stem().string() + ".png";
        cv::imwrite(output_filename, image);
    }
    return 0;
}
