
#include "iniio.h"
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
namespace iniio
{

int INI_BUF_SIZE = 2048;

IO::IO()
{
    flags_.push_back("#");
    flags_.push_back(";");
}

bool IO::parse(const string &content, string &key, string &value, char c/*= '='*/)
{
    int i = 0;
    int len = content.length();

    while (i < len && content[i] != c) {
        ++i;
    }

    if (i >= 0 && i < len) {
        key = string(content.c_str(), i);
        value = string(content.c_str() + i + 1, len - i - 1);
        return true;
    }

    return false;
}

int IO::getline(string &str, FILE *fp)
{
    int plen = 0;
    int buf_size = INI_BUF_SIZE * sizeof(char);

    char *buf = (char *) malloc(buf_size);
    char *pbuf = NULL;
    char *p = buf;

    if (buf == NULL) {
        fprintf(stderr, "no enough memory!exit!\n");
        exit(-1);
    }

    memset(buf, 0, buf_size);
    int total_size = buf_size;

    while (fgets(p, buf_size, fp) != NULL) {
        plen = strlen(p);

        if (plen > 0 && p[plen - 1] != '\n' && !feof(fp)) {

            total_size = strlen(buf) + buf_size;
            pbuf = (char *)realloc(buf, total_size);

            if (pbuf == NULL) {
                free(buf);
                fprintf(stderr, "no enough memory!exit!\n");
                exit(-1);
            }

            buf = pbuf;

            p = buf + strlen(buf);

            continue;
        } else {
            break;
        }
    }

    str = buf;

    free(buf);
    buf = NULL;
    return str.length();

}
int IO::load(const string &filename)
{
    release();
    fname_ = filename;
    SECTION *section = NULL;
    FILE *fp = fopen(filename.c_str(), "r");

    if (fp == NULL) {
        return -1;
    }

    string line;
    string comment;

    //增加默认段
    section = new SECTION();
    sections_[""] = section;

    while (getline(line, fp) > 0) {

        trimright(line, '\n');
        trimright(line, '\r');
        trim(line);

        if (!isComment(line)) {
            string subline;
            string tmp = line;

            for (size_t i = 0; i < flags_.size(); ++i) {
                subline = line.substr(0, line.find(flags_[i]));
                line = subline;
            }

            comment += tmp.substr(line.length());
        }

        trim(line);

        if (line.length() <= 0) {
            continue;
        }

        if (line[0] == '[') {
            section = NULL;
            int index = line.find_first_of(']');

            if (index == -1) {
                fclose(fp);
                fprintf(stderr, "unmatched ]\n");
                return -1;
            }

            int len = index - 1;

            if (len <= 0) {
                fprintf(stderr, "empty section\n");
                continue;
            }

            string s(line, 1, len);

            if (getSection(s.c_str()) != NULL) {
                fclose(fp);
                fprintf(stderr, "already existed section:%s\n", s.c_str());
                return -1;
            }

            section = new SECTION();
            sections_[s] = section;

            section->name = s;
            section->comment = comment;
            comment = "";
        } else if (isComment(line)) {
            if (comment != "") {
                comment += delim + line ;
            } else {
                comment = line;
            }
        } else {
            string key, value;

            if (parse(line, key, value)) {
                ITEM item;
                item.key = key;
                item.value = value;
                item.comment = comment;

                section->items.push_back(item);
            } else {
                fprintf(stderr, "failed to parse [%s]\n", line.c_str());
            }

            comment = "";
        }
    }

    fclose(fp);

    return 0;
}

int IO::save()
{
    return saveas(fname_);
}

int IO::saveas(const string &filename)
{
    string data = "";

    for (iterator sect = sections_.begin(); sect != sections_.end(); ++sect) {
        if (sect->second->comment != "") {
            data += sect->second->comment;
            data += delim;
        }

        if (sect->first != "") {
            data += string("[") + sect->first + string("]");
            data += delim;
        }

        for (SECTION::iterator item = sect->second->items.begin(); item != sect->second->items.end(); ++item) {
            if (item->comment != "") {
                data += item->comment;
                data += delim;
            }

            data += item->key + "=" + item->value;
            data += delim;
        }
    }

    FILE *fp = fopen(filename.c_str(), "w");

    fwrite(data.c_str(), 1, data.length(), fp);

    fclose(fp);

    return 0;
}
SECTION *IO::getSection(const string &section /*=""*/)
{
    iterator it = sections_.find(section);

    if (it != sections_.end()) {
        return it->second;
    }

    return NULL;
}

string IO::getStringValue(const string &section, const string &key, string def)
{
    string value, comment;

    int ret = getValue(section, key, value, comment);
	if(ret == ERR)
		value = def;
    return value;
}

int IO::getIntValue(const string &section, const string &key, int def)
{
    string value, comment;

    int ret = getValue(section, key, value, comment);
	if(ret != OK)
		return def;

    return atoi(value.c_str());
}

float IO::getFloatValue(const string &section, const string &key, float def)
{
    string value, comment;

    int ret = getValue(section, key, value, comment);
	if(ret != OK)
		return def;

    return (float)atof(value.c_str());

}

int IO::getValue(const string &section, const string &key, string &value)
{
    string comment;
    return getValue(section, key, value, comment);
}
int IO::getValue(const string &section, const string &key, string &value, string &comment)
{
    SECTION *sect = getSection(section);

    if (sect != NULL) {
        for (SECTION::iterator it = sect->begin(); it != sect->end(); ++it) {
            if (it->key == key) {
                value = it->value;
                comment = it->comment;
                return OK;
            }
        }
    }

    return ERR;
}
int IO::getValues(const string &section, const string &key, vector<string> &values)
{
    vector<string> comments;
    return getValues(section, key, values, comments);
}
int IO::getValues(const string &section, const string &key,
                       vector<string> &values, vector<string> &comments)
{
    string value, comment;

    values.clear();
    comments.clear();

    SECTION *sect = getSection(section);

    if (sect != NULL) {
        for (SECTION::iterator it = sect->begin(); it != sect->end(); ++it) {
            if (it->key == key) {
                value = it->value;
                comment = it->comment;

                values.push_back(value);
                comments.push_back(comment);
            }
        }
    }

    return (values.size() ? OK : ERR);

}
bool IO::hasSection(const string &section)
{
    return (getSection(section) != NULL);

}

bool IO::hasKey(const string &section, const string &key)
{
    SECTION *sect = getSection(section);

    if (sect != NULL) {
        for (SECTION::iterator it = sect->begin(); it != sect->end(); ++it) {
            if (it->key == key) {
                return true;
            }
        }
    }

    return false;
}
int IO::getSectionComment(const string &section, string &comment)
{
    comment = "";
    SECTION *sect = getSection(section);

    if (sect != NULL) {
        comment = sect->comment;
        return OK;
    }

    return ERR;
}
int IO::setSectionComment(const string &section, const string &comment)
{
    SECTION *sect = getSection(section);

    if (sect != NULL) {
        sect->comment = comment;
        return OK;
    }

    return ERR;
}

int IO::setValue(const string &section, const string &key,
                      const string &value, const string &comment /*=""*/)
{
    SECTION *sect = getSection(section);

    string comt = comment;

    if (comt != "") {
        comt = flags_[0] + comt;
    }

    if (sect == NULL) {
        sect = new SECTION();

        if (sect == NULL) {
            fprintf(stderr, "no enough memory!\n");
            exit(-1);
        }

        sect->name = section;
        sections_[section] = sect;
    }

    for (SECTION::iterator it = sect->begin(); it != sect->end(); ++it) {
        if (it->key == key) {
            it->value = value;
            it->comment = comt;
            return OK;
        }
    }

    //not found key
    ITEM item;
    item.key = key;
    item.value = value;
    item.comment = comt;

    sect->items.push_back(item);

    return OK;

}
void IO::getCommentFlags(vector<string> &flags)
{
    flags = flags_;
}
void IO::setCommentFlags(const vector<string> &flags)
{
    flags_ = flags;
}
void IO::deleteSection(const string &section)
{
    SECTION *sect = getSection(section);

    if (sect != NULL) {

        sections_.erase(section);
        delete sect;
    }
}
void IO::deleteKey(const string &section, const string &key)
{
    SECTION *sect = getSection(section);

    if (sect != NULL) {
        for (SECTION::iterator it = sect->begin(); it != sect->end(); ++it) {
            if (it->key == key) {
                sect->items.erase(it);
                break;
            }
        }
    }

}

void IO::release()
{
    fname_ = "";

    for (iterator i = sections_.begin(); i != sections_.end(); ++i) {
        delete i->second;
    }

    sections_.clear();

}

bool IO::isComment(const string &str)
{
    bool ret = false;

    for (size_t i = 0; i < flags_.size(); ++i) {
        size_t k = 0;

        if (str.length() < flags_[i].length()) {
            continue;
        }

        for (k = 0; k < flags_[i].length(); ++k) {
            if (str[k] != flags_[i][k]) {
                break;
            }
        }

        if (k == flags_[i].length()) {
            ret = true;
            break;
        }
    }

    return ret;
}
//for debug
void IO::print()
{
    printf("filename:[%s]\n", fname_.c_str());

    printf("flags_:[");

    for (size_t i = 0; i < flags_.size(); ++i) {
        printf(" %s ", flags_[i].c_str());
    }

    printf("]\n");

    for (iterator it = sections_.begin(); it != sections_.end(); ++it) {
        printf("section:[%s]\n", it->first.c_str());
        printf("comment:[%s]\n", it->second->comment.c_str());

        for (SECTION::iterator i = it->second->items.begin(); i != it->second->items.end(); ++i) {
            printf("    comment:%s\n", i->comment.c_str());
            printf("    parm   :%s=%s\n", i->key.c_str(), i->value.c_str());
        }
    }
}

void IO::trimleft(string &str, char c/*=' '*/)
{
    //trim head

    int len = str.length();

    int i = 0;

    while (str[i] == c && str[i] != '\0') {
        i++;
    }

    if (i != 0) {
        str = string(str, i, len - i);
    }
}

void IO::trimright(string &str, char c/*=' '*/)
{
    //trim tail
    int i = 0;
    int len = str.length();


    for (i = len - 1; i >= 0; --i) {
        if (str[i] != c) {
            break;
        }
    }

    str = string(str, 0, i + 1);
}

void IO::trim(string &str)
{
    //trim head

    int len = str.length();

	if(len < 1)
		return;

    int i = 0;

    while (isspace(str[i]) && str[i] != '\0') {
        i++;
    }

    if (i != 0) {
        str = string(str, i, len - i);
    }

    //trim tail
    len = str.length();

    for (i = len - 1; i >= 0; --i) {
        if (!isspace(str[i])) {
            break;
        }
    }

    str = string(str, 0, i + 1);
}
}
