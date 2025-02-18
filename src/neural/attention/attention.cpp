
#include "include/attention.hpp"

attention::attention(int tokenEmbed) : tokenEmbed(tokenEmbed) {
    head = std::vector<std::vector<double>>(tokenEmbed, std::vector<double>(tokenEmbed, 0));
    MQ = mat(tokenEmbed, tokenEmbed);
    MK = mat(tokenEmbed, tokenEmbed);
    MV = mat(tokenEmbed, tokenEmbed);
    MH = mat(tokenEmbed, tokenEmbed);
    dH = std::vector<std::vector<double>>(tokenEmbed, std::vector<double>(tokenEmbed, 0));
    dV = std::vector<std::vector<double>>(tokenEmbed, std::vector<double>(tokenEmbed, 0));
    E1 = std::vector<double>(tokenEmbed, 0);
    E2 = std::vector<double>(tokenEmbed, 0);
    d1 = std::vector<double>(tokenEmbed, 0);
    d2 = std::vector<double>(tokenEmbed, 0);
    v = mlp(tokenEmbed, tokenEmbed);
    h = mlp(tokenEmbed, tokenEmbed);
}
