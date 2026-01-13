print_parent_oid(FILE * f, struct tree *tp)
{
    if (tp) {
        if (tp->parent) {
            print_parent_oid(f, tp->parent);
         /*RECURSE*/}
        fprintf(f, ".%lu", tp->subid);
    }
}