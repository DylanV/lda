#include "gtest/gtest.h"
#include "data.h"

TEST(SplitTest, TwoWords) {
    std::vector<std::string> items = split("first second", ' ');
    ASSERT_EQ(items.size(), 2);
    ASSERT_EQ(items[0], "first");
    ASSERT_EQ(items[1], "second");
}

TEST(SplitTest, ThreeWordsNonSpaceDelim) {
    std::vector<std::string> items = split("first-second-third third", '-');
    ASSERT_EQ(items.size(), 3);
    ASSERT_EQ(items[0], "first");
    ASSERT_EQ(items[1], "second");
    ASSERT_EQ(items[2], "third third");
}

TEST(SettingsLoadTest, LoadMockFile) {
    std::map<std::string, std::string> items = load_settings_file("../test/test_files/mock_settings.dat");

    ASSERT_EQ(items.size(), 3);
    ASSERT_EQ(items["FIRST"], "first");
    ASSERT_EQ(items["SECOND"], "2");
    ASSERT_EQ(items["THIRD"], "true");
}