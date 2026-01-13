int sgwu_gtp_open(void)
{
    ogs_socknode_t *node = NULL;
    ogs_sock_t *sock = NULL;

    ogs_list_for_each(&ogs_gtp_self()->gtpu_list, node) {
        sock = ogs_gtp_server(node);
        if (!sock) return OGS_ERROR;

        if (sock->family == AF_INET)
            ogs_gtp_self()->gtpu_sock = sock;
        else if (sock->family == AF_INET6)
            ogs_gtp_self()->gtpu_sock6 = sock;

        node->poll = ogs_pollset_add(ogs_app()->pollset,
                OGS_POLLIN, sock->fd, _gtpv1_u_recv_cb, sock);
        ogs_assert(node->poll);
    }

    OGS_SETUP_GTPU_SERVER;

    return OGS_OK;
}