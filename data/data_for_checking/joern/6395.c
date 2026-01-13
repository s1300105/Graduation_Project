static LUA_FUNCTION(openssl_x509_export)
{
  X509 *cert = CHECK_OBJECT(1, X509, "openssl.x509");
  int fmt = luaL_checkoption(L, 2, "pem", format);
  int notext = lua_isnone(L, 3) ? 1 : lua_toboolean(L, 3);
  BIO* out = NULL;

  if (fmt != FORMAT_DER && fmt != FORMAT_PEM)
  {
    luaL_argerror(L, 2, "format only accept pem or der");
  }

  out  = BIO_new(BIO_s_mem());
  if (fmt == FORMAT_PEM)
  {
    if (!notext)
    {
      X509_print(out, cert);
    }

    if (PEM_write_bio_X509(out, cert))
    {
      BUF_MEM *bio_buf;
      BIO_get_mem_ptr(out, &bio_buf);
      lua_pushlstring(L, bio_buf->data, bio_buf->length);
    }
    else
      lua_pushnil(L);
  }
  else
  {
    if (i2d_X509_bio(out, cert))
    {
      BUF_MEM *bio_buf;
      BIO_get_mem_ptr(out, &bio_buf);
      lua_pushlstring(L, bio_buf->data, bio_buf->length);
    }
    else
      lua_pushnil(L);
  }

  BIO_free(out);
  return 1;
};