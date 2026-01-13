print_parent_labeledoid(FILE * f, struct tree *tp)
{
    if (tp) {
        if (tp->parent) {
            print_parent_labeledoid(f, tp->parent);
         /*RECURSE*/}
        fprintf(f, ".%s(%lu)", tp->label, tp->subid);
    }
}