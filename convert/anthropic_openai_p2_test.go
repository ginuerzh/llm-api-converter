package convert

import (
	"fmt"
	"sync"
	"testing"
)

func TestP2_ReasoningCacheConcurrentGetPut(t *testing.T) {
	cache := NewReasoningCache(16)
	defer cache.Close()

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(2)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < 2000; j++ {
				cache.Put([]string{"tool_1"}, fmt.Sprintf("reasoning-%d-%d", i, j))
			}
		}(i)
		go func() {
			defer wg.Done()
			for j := 0; j < 2000; j++ {
				cache.Get([]string{"tool_1"})
			}
		}()
	}
	wg.Wait()
}
