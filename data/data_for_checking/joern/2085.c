void smf_gtp_close(void)
{
    if (ogs_gtp_self()->link_local_addr)
        ogs_freeaddrinfo(ogs_gtp_self()->link_local_addr);

    ogs_socknode_remove_all(&ogs_gtp_self()->gtpc_list);
    ogs_socknode_remove_all(&ogs_gtp_self()->gtpc_list6);

    ogs_socknode_remove_all(&ogs_gtp_self()->gtpu_list);
}