#pragma once
#include <string_view>
#include <unordered_map>
#include <string>
#include <sstream>
#include <utility>
#include <iostream> 

using namespace std;

namespace Ini
{

using Section = unordered_map<string, string>;

class Document {
public:
  Section& AddSection(string name);
  const Section& GetSection(const string& name) const;
  size_t SectionCount() const;
    unordered_map<string, Section> sections;

private:
  //unordered_map<string, Section> sections;
};
pair<string_view, string_view> Split(string_view line, char by);
string_view Lstrip(string_view line);
Document Load(istream& input);

}
