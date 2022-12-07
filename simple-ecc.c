#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

typedef struct {
    int64_t x;
    int64_t y;
} EccPoint;

typedef struct {
    int64_t a;
    int64_t b;
    int64_t p;
} EC; // y^2 = x^3 + ax + b (mod p)

typedef struct {
    EccPoint pt;
    int64_t n;
} GenPoint;

typedef struct {
    EC ec;
    GenPoint genpt;
} ECDLP;

int64_t modp(int64_t x, int64_t p)
{
    while (x < 0) x += p;
    x %= p;
    return x;
}

int64_t inverse(int64_t x, int64_t p)
{
    int64_t q = 0;
    int64_t r = 0;

    int64_t r1 = p;
    int64_t r2 = x;

    int64_t t1 = 0;
    int64_t t2 = 1;
    int64_t t = 0;
    while (r2 > 0)
    {
        q = r1 / r2, r = r1 % r2;
        r1 = r2, r2 = r;

        t = t1 - q * t2;
        t1 = t2, t2 = t;
    }

    t1 = modp(t1, p);
    return t1;
}

void addpt(
    const EC* ec,
    const EccPoint* pt1, const EccPoint* pt2,
    EccPoint* new_pt
)
{
    if (pt1->x == -1 && pt1->y == -1)
    {
        new_pt->x = pt2->x;
        new_pt->y = pt2->y;
        return;
    }
    else if (pt2->x == -1 && pt2->y == -1)
    {
        new_pt->x = pt1->x;
        new_pt->y = pt1->y;
        return;
    }
    else
    {
        int64_t lambda = 0;
        int64_t new_x = 0;
        int64_t new_y = 0;

        if (pt1->x == pt2->x)
        {
            // Unit
            if (pt1->y + pt2->y == ec->p)
            {
                new_pt->x = -1;
                new_pt->y = -1;
                return;
            }
            // Same
            else if (pt1->y == pt2->y)
            {
                lambda = (3 * pt1->x * pt1->x + ec->a) * inverse(2 * pt1->y, ec->p);
            }
            else
            {
                exit(-1);
            }
        }
        // Different
        else
        {
            int64_t delta_x = 0;
            int64_t delta_y = 0;

            delta_x = modp((pt2->x - pt1->x), ec->p);
            delta_y = modp((pt2->y - pt1->y), ec->p);
            lambda = delta_y * inverse(delta_x, ec->p);
        }

        lambda %= ec->p;

        new_x = modp((lambda * lambda - pt1->x - pt2->x), ec->p);
        new_y = modp((lambda * (pt1->x - new_x) - pt1->y), ec->p);

        new_pt->x = new_x;
        new_pt->y = new_y;
        return;
    }
    return;
}

void mulpt(
    const EC* ec,
    uint64_t k, const EccPoint* pt,
    EccPoint* new_pt
)
{
    if (k == 0)
    {
        new_pt->x = -1;
        new_pt->y = -1;
        return;
    }

    int i = 64;
    // find first 1 bit
    while (!(k & 0x8000000000000000))
    {
        k <<= 1;
        i--;
    }

    EccPoint ret = { pt->x, pt->y };
    EccPoint misc = { pt->x, pt->y };

    k <<= 1;
    i--;
    while (i > 0)
    {
        addpt(ec, &ret, &ret, &ret);
        if (k & 0x8000000000000000)
        {
            addpt(ec, &ret, pt, &ret);
        }
        else
        {
            addpt(ec, &ret, pt, &misc);
        }
        k <<= 1;
        i--;
    }

    if (&ret && &misc)
    {
        new_pt->x = ret.x;
        new_pt->y = ret.y;
    }

    return;
}

uint8_t isprime(int64_t x)
{
    for (int64_t i = 2; i < (int64_t)sqrtl((long double)x); i++)
    {
        if (x % i == 0)
        {
            return 0;
        }
    }
    return 1;
}

int64_t compute_ptrank(const EC* ec, const EccPoint* pt)
{
    EccPoint tmp = { pt->x, pt->y };

    int64_t rank = 1;
    while (1)
    {
        addpt(ec, &tmp, pt, &tmp);
        if (tmp.x == pt->x && tmp.y == pt->y)
        {
            break;
        }
        rank++;
    }
    return rank;
}

void print_points(EC* ec)
{
    printf("EC: { a = %lld, b = %lld, p = %lld }\n", ec->a, ec->b, ec->p);
    if (modp(4 * ec->a * ec->a * ec->a + 27 * ec->b * ec->b, ec->p) == 0)
    {
        printf("Params Invalid!\n");
        return;
    }
    else
    {
        int64_t pt_count = 0;
        int64_t y_sqr = 0;
        int64_t y = 0;
        uint8_t find = 0;
        EccPoint pt_tmp = { 0 };
        int64_t pt_rank = 0;
        char prime_rank = 0;
        for (int64_t x = 0; x < ec->p; x++)
        {
            y_sqr = modp((x * x * x + ec->a * x + ec->b), ec->p);
            find = 0;
            while (y_sqr < (ec->p - 1) * (ec->p - 1))
            {
                y = (int64_t)sqrtl((long double)y_sqr);
                if (y * y == y_sqr)
                {
                    find = 1;
                    break;
                }
                y_sqr += ec->p;
            }
            if (find)
            {
                pt_tmp.x = x;
                pt_tmp.y = y;
                pt_rank = compute_ptrank(ec, &pt_tmp);
                prime_rank = (isprime(pt_rank) ? 'P' : 'C');
                pt_count++;
                printf("(%5lld, %5lld)[%5lld][%c], ", pt_tmp.x, pt_tmp.y, pt_rank, prime_rank);
                if (pt_count % 4 == 0)
                {
                    printf("\n");
                }

                if (y != 0)
                {
                    pt_tmp.y = ec->p - y;
                    pt_count++;
                    printf("(%5lld, %5lld)[%5lld][%c], ", pt_tmp.x, pt_tmp.y, pt_rank, prime_rank);
                    if (pt_count % 4 == 0)
                    {
                        printf("\n");
                    }
                }
            }
        }
        printf("(%5d, %5d)[%5d][%c]\n", -1, -1, 1, 'C');
        pt_count++;
        printf("EccPoints Count: %lld\n", pt_count);
    }

    return;
}

void encrypt_blk(
    const ECDLP* ecdlp,
    const EccPoint* pubkey,
    int64_t plain,
    EccPoint* cipher_pt, int64_t* cipher
)
{
    int64_t rndk = 0;
    EccPoint x2 = { 0 };
    do
    {
        rndk = rand() % (ecdlp->genpt.n - 1) + 1;
        mulpt(&ecdlp->ec, rndk, pubkey, &x2);
    } while (x2.x == 0);

    mulpt(&ecdlp->ec, rndk, &ecdlp->genpt.pt, cipher_pt);
    *cipher = (plain * x2.x) % ecdlp->genpt.n;

    return;
}

void decrypt_blk(
    const ECDLP* ecdlp,
    int64_t prikey,
    const EccPoint* cipher_pt, int64_t cipher,
    int64_t* plain
)
{
    EccPoint x2 = { 0 };
    mulpt(&ecdlp->ec, prikey, cipher_pt, &x2);
    *plain = (cipher * inverse(x2.x, ecdlp->genpt.n)) % ecdlp->genpt.n;

    return;
}

int main()
{
    srand((uint32_t)time(NULL));
    ECDLP ecdlp = { { 2, 11, 49177 }, {{1, 14445}, 49031} };
    print_points(&ecdlp.ec);

    printf("Curve Params: { a = %lld, b = %lld, p = %lld }\n", ecdlp.ec.a, ecdlp.ec.b, ecdlp.ec.p);
    printf("GenPoint: { pt: (%lld, %lld), n: %lld }\n", ecdlp.genpt.pt.x, ecdlp.genpt.pt.y, ecdlp.genpt.n);

    int64_t prikey = 149;
    EccPoint pubkey = { 0 };
    mulpt(&ecdlp.ec, prikey, &ecdlp.genpt.pt, &pubkey);
    printf("Prikey: %lld Pubkey: (%lld, %lld)\n", prikey, pubkey.x, pubkey.y);

    int64_t plain = 23456;
    int64_t cipher = -1;
    EccPoint cipher_pt = { 0 };

    printf("Plain: %lld\n", plain);

    encrypt_blk(&ecdlp, &pubkey, plain, &cipher_pt, &cipher);
    printf("Cipher: (%lld, %lld), %lld\n", cipher_pt.x, cipher_pt.y, cipher);

    decrypt_blk(&ecdlp, prikey, &cipher_pt, cipher, &plain);
    printf("Plain: %lld\n", plain);

    return 0;
}
