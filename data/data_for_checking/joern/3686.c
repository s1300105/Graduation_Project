snprint_octet_string(char *buf, size_t buf_len,
                     const netsnmp_variable_list * var, const struct enum_list *enums,
                     const char *hint, const char *units)
{
    size_t          out_len = 0;
    if (sprint_realloc_octet_string
        ((u_char **) & buf, &buf_len, &out_len, 0, var, enums, hint,
         units))
        return (int) out_len;
    else
        return -1;
}