class Solution {

public:

    char arr[50001];

    int n;

    void move_left(int &l) {

        while (arr[l] == arr[l+1] && l+1 <= n) l++;

    }

    int lengthOfLongestSubstring(string s) {

        n = s.length();

        strcpy(arr+1, s.c_str());

        int l = 1;

        int r;

        bool hit_end = false;

        int ans = 0;

        while (!hit_end) {

            // init vis array

            vector<bool> vis(256, false);

            // move left

            move_left(l);

            // record left

            vis[static_cast<unsigned char>(arr[l])] = true;

            // record ans

            ans = max(ans, 1);

            // update right, need to perform termination check

            if (l + 1 <= n) r = l + 1;

            else {

                hit_end = true;

                break;

            }

            // record right

            vis[static_cast<unsigned char>(arr[r])] = true;

            ans = max(ans, r - l + 1);

            // move right

            while (!hit_end) {

                r++;

                if (r > n) {

                    hit_end = true;

                    break;

                }

                // if not hit end, check if hit dup

                if (vis[static_cast<unsigned char>(arr[r])]) {

                    // if hit dup, update l = r and go to next round.

                    l = r;

                    break;    

                } else {

                    // if not hit dup, record right

                    vis[static_cast<unsigned char>(arr[r])] = true;

                    // record ans

                    ans = max(ans, r - l + 1);

                }

            }

        }

        return ans;

    }

};