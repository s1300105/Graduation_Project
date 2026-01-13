SproutNote SproutNotePlaintext::note(const SproutPaymentAddress& addr) const
{
    return SproutNote(addr.a_pk, value_, rho, r);
}