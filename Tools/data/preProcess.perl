#!/usr/bin/perl -l

# Program to filter Wikipedia XML dumps to "clean" text consisting only of lowercase
# letters (a-z, converted from A-Z), and spaces (never consecutive).  
# All other characters are converted to spaces.  Only text which normally appears 
# in the web browser is displayed.  Tables are removed.  Image captions are 
# preserved.  Links are converted to normal text.  Digits are spelled out.
# Written by Matt Mahoney, June 10, 2006.  This program is released to the public domain.

$/="\n";                     # input record separator
while (<>) {
    # Remove any text not normally visible
    s/&amp;/&/g;            # decode URL encoded chars
    s/&lt;/</g;
    s/&gt;/>/g;
    s/<[^>]*>//g;           # remove xhtml tags
    s/\[http:[^] ]*/[/g;    # remove normal url, preserve visible text
    s/\|thumb//ig;          # remove images links, preserve caption
    s/\|left//ig;
    s/\|right//ig;
    s/\|\d+px//ig;
    s/\[\[image:[^\[\]]*\|//ig;
    s/\[\[category:([^|\]]*)[^]]*\]\]/[[$1]]/ig;  # show categories without markup
    s/\[\[[a-z\-]*:[^\]]*\]\]//g;  # remove links to other languages
    # s/\{?\{[^\{]*?\|?([^\{\|]+?)\}\}?/$1/ig;
    # s/\[?\[[^\[]*?\|?([^\[\|]+?)\]\]?/$1/ig;
    # s/\[\[[^\|\]]*\|/[[/g;  # remove wiki url, preserve visible text
    # s/\{\{[^\}]*\}\}//g;         # remove {{icons}} and {tables}
    # s/\{[^\}]*\}//g;
    s/&[^;]*;/ /g;          # remove URL encoded chars
    s/\W+/ /g;
    chop;
    print $_;
}
