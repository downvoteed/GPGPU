#pragma once
#include <iostream>
#include <thread>
#include <future>

#if defined __linux__
// Linux
#include <sys/resource.h>
#include <unistd.h>
#include <stdio.h>

size_t get_current_rss()
{
	static size_t page_size = sysconf(_SC_PAGESIZE);

	FILE* stat_file = fopen("/proc/self/statm", "r");
	if (!stat_file) return 0;

	size_t pages_count = 0;
	fscanf(stat_file, "%ld %ld", &pages_count);
	fclose(stat_file);

	// Compute the size in bytes.
	return pages_count * page_size;
}

size_t get_peak_rss()
{
	rusage usage_data;
	getrusage(RUSAGE_SELF, &usage_data);

	return size_t(usage_data.ru_maxrss) * 1024;
}

#endif
#if defined __APPLE__
// OSX
#include <sys/resource.h>
#include <unistd.h>
#include <mach/mach.h>

size_t get_current_rss()
{
	mach_task_basic_info info;
	mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;

	kern_return_t result = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
		(task_info_t)&info, &count);
	if (result != KERN_SUCCESS) return 0;

	return size_t(info.resident_size);
}

size_t get_peak_rss()
{
	rusage usage_data;
	getrusage(RUSAGE_SELF, &usage_data);

	return size_t(usage_data.ru_maxrss);
}

#endif


#if defined _WIN32

// Windows
#include <windows.h>
#include <psapi.h>

size_t get_current_rss()
{
	static HANDLE process = GetCurrentProcess();

	PROCESS_MEMORY_COUNTERS counters;
	GetProcessMemoryInfo(process, &counters, sizeof(PROCESS_MEMORY_COUNTERS));
	return counters.WorkingSetSize;
}


size_t get_peak_rss()
{
	static HANDLE process = GetCurrentProcess();

	PROCESS_MEMORY_COUNTERS counters;
	GetProcessMemoryInfo(process, &counters, sizeof(PROCESS_MEMORY_COUNTERS));
	return counters.PeakWorkingSetSize;
}

#endif

// Often it's not convenient to report memory usage in bytes, numbers are too big
// and unreadable. Convert to MB.
double toMB(size_t size_in_bytes)
{
	return double(size_in_bytes) / (1024.0 * 1024.0);
}
class LogRss
{

public:
	std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
	std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
	std::shared_ptr<std::thread> memory_thread;
	std::promise<void> exit_signal;
	int seconds = -1;
	/**
		* @brief Launch loging of rss in a new thread. The current rss is
		* displayed every second.
		*
		* @param function_name
		*/
	void begin_rss_loging()
	{
		memory_thread = std::make_shared<std::thread>(([&] {
			std::future<void> exit_condition = exit_signal.get_future();

			printf("initial rss %.2f GB\n", toMB(get_current_rss()));

			double previous_rss = 0;
			while (exit_condition.wait_for(std::chrono::milliseconds(1000))
				== std::future_status::timeout)
			{
				// If there is a change in the resident set, log it.
				double current_rss = toMB(get_current_rss());
				if (current_rss != previous_rss)
				{
					previous_rss = current_rss;

					this->seconds++;

					printf(" - ");
					printf("rss %.4f MB\n", current_rss);
				}
			}
			printf("final rss %.2f MB\n", toMB(get_current_rss()));
			}));
		start_time = std::chrono::high_resolution_clock::now();
	}
	/**
	 * @brief Display time completion and set the exit signal to finish the
	 * thread.
	 *
	 */
	void end_rss_loging()
	{
		auto end_time = std::chrono::high_resolution_clock::now();
		size_t seconds = std::chrono::duration_cast<std::chrono::seconds>(
			end_time - start_time)
			.count();

		exit_signal.set_value();
		memory_thread->join();
		printf("Time to completion %.2f sec\n", double(seconds) / 1000.0);
	}
};
