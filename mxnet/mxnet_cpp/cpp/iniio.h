#pragma once
#include <map>
#include <vector>
#include <string>
using namespace std;
namespace iniio
{
const int OK  = 0;
const int ERR = -1;
const string delim = "\n";
struct ITEM {
    string key;
    string value;
    string comment;
};
struct SECTION {
    typedef vector<ITEM>::iterator iterator;
    iterator begin() {
        return items.begin();
    }
    iterator end() {
        return items.end();
    }

    string name;
    string comment;
    vector<ITEM> items;
};

class IO
{
public:
    IO();
    ~IO() {
        release();
    }

public:
    typedef map<string, SECTION *>::iterator iterator;

    iterator begin() {
        return sections_.begin();
    }
    iterator end() {
        return sections_.end();
    }
public:
    int load(const string &fname);
    int save();
    int saveas(const string &fname);

    string getStringValue(const string &section, const string &key, string def="");
    int getIntValue(const string &section, const string &key, int def=0);
    float getFloatValue(const string &section, const string &key, float def=0.0);

    int getValue(const string &section, const string &key, string &value);
    int getValue(const string &section, const string &key, string &value, string &comment);

    int getValues(const string &section, const string &key, vector<string> &values);
    int getValues(const string &section, const string &key, vector<string> &value, vector<string> &comments);

    bool hasSection(const string &section) ;
    bool hasKey(const string &section, const string &key) ;

    int getSectionComment(const string &section, string &comment);
    int setSectionComment(const string &section, const string &comment);
    void getCommentFlags(vector<string> &flags);
    void setCommentFlags(const vector<string> &flags);

    int setValue(const string &section, const string &key, const string &value, const string &comment = "");
    void deleteSection(const string &section);
    void deleteKey(const string &section, const string &key);
public:
    static void trimleft(string &str, char c = ' ');
    static void trimright(string &str, char c = ' ');
    static void trim(string &str);
private:
    SECTION *getSection(const string &section = "");
    void release();
    int getline(string &str, FILE *fp);
    bool isComment(const string &str);
    bool parse(const string &content, string &key, string &value, char c = '=');
    //for dubug
    void print();

private:
    map<string, SECTION *> sections_;
    string fname_;
    vector<string> flags_;
};

}

