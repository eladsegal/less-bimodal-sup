// Modified from http://www.kernel.org/doc/man-pages/online/pages/man2/getdents.2.html
// According to http://be-n.com/spw/you-can-list-a-million-files-in-a-directory-but-not-with-ls.html
// Compile with gcc tools/fast_listdir.c -o tools/fast_listdir
// Run with tools/fast_listdir PATH

#define _GNU_SOURCE
#include <dirent.h>     /* Defines DT_* constants */
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/syscall.h>

#define handle_error(msg) \
        do { perror(msg); exit(EXIT_FAILURE); } while (0)

struct linux_dirent {
    unsigned long  d_ino;
    off_t          d_off;
    unsigned short d_reclen;
    char           d_name[];
};

#define BUF_SIZE 1024*1024*5

int
main(int argc, char *argv[])
{
    int fd;
    long nread;
    long files_counter;
    long getdents_counter;
    time_t start, end;
    double time_passed;
    char buf[BUF_SIZE];
    struct linux_dirent *d;
    char d_type;
    char target_d_type;

    if (argc >= 3) {
        if (argv[2][0] == 'd') {
            target_d_type = DT_DIR;
        } else if (argv[2][0] == 'f') {
            target_d_type = DT_REG;
        } else if (argv[2][0] == 'u') {
            target_d_type = DT_UNKNOWN;
        }
    } else {
        printf("Usage: %s PATH d|f\n(d=dir, f=file)\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    fd = open(argc > 1 ? argv[1] : ".", O_RDONLY | O_DIRECTORY);
    if (fd == -1)
        handle_error("open");

    files_counter = 0;
    start = time(NULL);
    for (;;) {
        nread = syscall(SYS_getdents, fd, buf, BUF_SIZE);
        getdents_counter += 1;
        if (nread == -1)
            handle_error("getdents");

        if (nread == 0)
            break;

        for (long bpos = 0; bpos < nread;) {
            d = (struct linux_dirent *) (buf + bpos);
            if(d->d_ino) {
                d_type = *(buf + bpos + d->d_reclen - 1);
                if ((d_type == target_d_type) && (strcmp(d->d_name, ".") != 0 && strcmp(d->d_name, "..") != 0)) {
                    printf("%s\n", (char *) d->d_name);
                    files_counter++;
                }
            }
            bpos += d->d_reclen;
        }
        // print to stderr on the same line
        end = time(NULL);
        time_passed = (double) (end - start);
        fprintf(stderr, "\rgetdents calls: %ld, files read: %ld, time (seconds): %.2f, time (minutes): %.2f", getdents_counter, files_counter, time_passed, time_passed / 60);
        fflush(stderr);
    }
    fprintf(stderr, "\n");
}
