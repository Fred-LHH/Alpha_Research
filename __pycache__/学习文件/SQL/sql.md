inner join：2表值都存在

outer join：附表中值可能存在null的情况。

总结：

　　①A inner join B：取交集

　　②A left join B：取A全部，B没有对应的值，则为null

　　③A right join B：取B全部，A没有对应的值，则为null

　　④A full outer join B：取并集，彼此没有对应的值为null

　　上述4种的对应条件，在on后填写。
