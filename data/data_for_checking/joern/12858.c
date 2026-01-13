scan_objlist(struct node *root, struct module *mp, struct objgroup *list, const char *error)
{
    int             oLine = mibLine;

    while (list) {
        struct objgroup *gp = list;
        struct node    *np;
        list = list->next;
        np = root;
        while (np)
            if (label_compare(np->label, gp->name))
                np = np->next;
            else
                break;
        if (!np) {
	    int i;
	    struct module_import *mip;
	    /* if not local, check if it was IMPORTed */
	    for (i = 0, mip = mp->imports; i < mp->no_imports; i++, mip++)
		if (strcmp(mip->label, gp->name) == 0)
		    break;
	    if (i == mp->no_imports) {
		mibLine = gp->line;
		print_error(error, gp->name, QUOTESTRING);
	    }
        }
        free(gp->name);
        free(gp);
    }
    mibLine = oLine;
}