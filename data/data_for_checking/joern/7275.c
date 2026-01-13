print_ascii_dump_tree(FILE * f, struct tree *tree, int count)
{
    struct tree    *tp;

    count++;
    for (tp = tree->child_list; tp; tp = tp->next_peer) {
        fprintf(f, "%s OBJECT IDENTIFIER ::= { %s %ld }\n", tp->label,
                tree->label, tp->subid);
    }
    for (tp = tree->child_list; tp; tp = tp->next_peer) {
        if (tp->child_list)
            print_ascii_dump_tree(f, tp, count);
    }
}