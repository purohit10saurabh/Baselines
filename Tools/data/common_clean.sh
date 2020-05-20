
get_lines(){
	wc -l $1 | awk -F ' ' '{print $1}'
}

ft_lb=$(python remove_invalid.py $1 $1'.temp')
echo $(get_lines $1'.temp') $ft_lb > $2
cat $1'.temp' >> $2
rm -rf $1.'temp'